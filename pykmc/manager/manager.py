from .session import Session
from concurrent.futures import Future
from dataclasses import dataclass, field
import collections
import queue
import threading
from typing import Any, Literal


@dataclass
class Job:
    """Unit of work dispatched to a thread worker.

    Parameters
    ----------
    op_name : str
        Operation name in the Worker registry.
    kwargs : dict
        Keyword arguments forwarded to the operation.
    future : Future
        Resolved with the result once the job completes.
    """

    op_name: str
    kwargs: dict = field(default_factory=dict)
    future: Future = field(default_factory=Future)


class Manager:
    def __init__(
        self,
        local_sessions: list[Session],
        global_session: Session | None = None,
        group_session: Session | None = None,
    ) -> None:
        self.local_sessions = local_sessions
        self.global_session = global_session
        self.group_session = group_session

        self._local_queue: queue.Queue[Job] = queue.Queue()
        self._local_threads: list[threading.Thread] = []

        self.mode: Literal["local", "group", "global"] = "local"
        # Operations submitted since the start, by name (all modes); the KMC loop
        # reports the per-step differences (searches, refinements, prefactors)
        self.op_counts: collections.Counter[str] = collections.Counter()

    def start(self) -> None:
        """Start one thread per local session. Must be called before submit().

        Raises
        ------
        RuntimeError
            If start() has already been called.
        """
        if self._local_threads:
            raise RuntimeError("Manager is already started.")
        for session in self.local_sessions:
            t = threading.Thread(
                target=self._worker_loop, args=(session, self._local_queue), daemon=True
            )
            t.start()
            self._local_threads.append(t)

    def shutdown(self) -> None:
        """Stop all thread workers and shut down sessions."""
        if self.mode != "local":
            self._use_local()
        for _ in self._local_threads:
            self._local_queue.put(None)
        for t in self._local_threads:
            t.join()
        self._local_threads.clear()
        for session in self.local_sessions:
            session.shutdown()

    def list_ops(self, mode: str = "local") -> list[str]:
        """Return the list of available operations for the given mode."""
        return self.submit("list_ops", mode=mode).result()

    def broadcast(self, op_name: str, **kwargs) -> None:
        """Send the same op to all local sessions sequentially.

        Useful for initialisation steps that every worker must run.
        Switches to local mode automatically if needed.
        """
        if self.mode != "local":
            self._use_local()
        self._local_queue.join()
        for session in self.local_sessions:
            session.call(op_name, **kwargs)

    # ------------------------------------------------------------------
    # Mode transitions
    # ------------------------------------------------------------------

    def _use_local(self) -> None:
        """Switch all workers back to local mode.

        Only one message is sent (to the global or group session master rank).
        All workers in that communicator receive it via the bcast in Worker._loop,
        so a single send is enough to switch the entire collective back to local.
        """
        if self.mode == "global":
            self.global_session.use_local()
        elif self.mode == "group":
            self.group_session.use_local()
        self.mode = "local"

    def _use_global(self) -> None:
        """Drain local queue then switch all workers to global mode."""
        if self.mode != "local":
            self._use_local()
        self._local_queue.join()
        for session in self.local_sessions:
            session.use_global()
        self.mode = "global"

    def _use_group(self) -> None:
        """Drain local queue then switch workers to group mode.

        Workers without a group_comm silently stay in local mode and become
        idle for the duration of the group operation.
        """
        if self.mode != "local":
            self._use_local()
        self._local_queue.join()
        for session in self.local_sessions:
            session.use_group()
        self.mode = "group"

    # ------------------------------------------------------------------
    # Job submission
    # ------------------------------------------------------------------

    def _worker_loop(self, session: Session, job_queue: queue.Queue) -> None:
        """Pull jobs from the queue and execute via the session."""
        while True:
            job = job_queue.get()
            if job is None:  # sentinel — stop
                break
            try:
                result = session.call(job.op_name, **job.kwargs)
                job.future.set_result(result)
            except Exception as e:
                job.future.set_exception(e)
            finally:
                job_queue.task_done()

    def submit(self, op_name: str, **kwargs) -> Future:
        """Submit a job to the local worker pool (async).

        If the manager is currently in group or global mode, switches back to
        local mode first. Similarly, submit_group() and submit_global() switch
        to their respective modes automatically before dispatching.

        Returns
        -------
        Future
            Resolved when the job completes.

        Raises
        ------
        RuntimeError
            If start() has not been called.
        """
        if not self._local_threads:
            raise RuntimeError("Manager has not been started. Call start() first.")
        if self.mode != "local":
            self._use_local()
        job = Job(op_name=op_name, kwargs=kwargs)
        self.op_counts[op_name] += 1
        self._local_queue.put(job)
        return job.future

    def submit_group(self, op_name: str, **kwargs) -> Any:
        """Submit a job to the group worker and block until it completes.

        The group spans only the subset of workers configured at factory time.
        All other workers are idle for the duration of the call.

        Returns
        -------
        Any
            Result of the operation, or None for void operations.

        Raises
        ------
        RuntimeError
            If no group session was configured.
        """
        if self.group_session is None:
            raise RuntimeError("No group session configured.")
        if self.mode != "group":
            self._use_group()
        self.op_counts[op_name] += 1
        return self.group_session.call(op_name, **kwargs)

    def submit_global(self, op_name: str, **kwargs) -> Any:
        """Submit a job to the global worker and block until it completes.

        All MPI ranks work collectively, so the call is synchronous.

        Returns
        -------
        Any
            Result of the operation, or None for void operations.

        Raises
        ------
        RuntimeError
            If no global session was configured.
        """
        if self.global_session is None:
            raise RuntimeError("No global session configured.")
        if self.mode != "global":
            self._use_global()
        self.op_counts[op_name] += 1
        return self.global_session.call(op_name, **kwargs)

    def __getattr__(self, name: str):
        """Auto-generate submit wrappers from attribute access.

        mgr.minimize(positions=pos)        → submit("minimize", positions=pos)        → Future
        mgr.group_minimize(positions=pos)  → submit_group("minimize", positions=pos)  → result
        mgr.global_minimize(positions=pos) → submit_global("minimize", positions=pos) → result
        """
        if name.startswith("global_"):
            op = name[len("global_") :]
            return lambda **kw: self.submit_global(op, **kw)
        if name.startswith("group_"):
            op = name[len("group_") :]
            return lambda **kw: self.submit_group(op, **kw)
        return lambda **kw: self.submit(name, **kw)

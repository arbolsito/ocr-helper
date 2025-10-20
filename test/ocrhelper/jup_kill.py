
import os


if os.name == "nt":
    print("Interrupt message not supported on Windows")
else:
    pid = os.getpid()
    pgid = os.getpgid(pid)
    # Prefer process-group over process
    # but only if the kernel is the leader of the process group
    if pgid and pgid == pid and hasattr(os, "killpg"):
        try:
            os.killpg(pgid, SIGINT)
        except OSError:
            os.kill(pid, SIGINT)
            raise
    else:
        os.kill(pid, SIGINT)
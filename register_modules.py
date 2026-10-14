import ultralytics.nn.tasks as tasks
import ultralytics.nn.modules as modules
from custom_modules import LWGA


def register():
    tasks.LWGA = LWGA
    modules.LWGA = LWGA
    print("[Info] LWGA registered.")
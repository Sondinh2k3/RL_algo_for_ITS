# worker_interface_demo.py
import ray
import time
from abc import ABC, abstractmethod
from collections import defaultdict

# ==========================================================
# EVENT BUS: trung tâm truyền thông tin giữa các group
# ==========================================================
@ray.remote
class EventBus:
    def __init__(self):
        # group_name -> [(worker_id, actor_ref)]
        self.groups = defaultdict(list)

    def subscribe(self, group_name: str, worker_ref, worker_id: str):
        """Đăng ký worker vào group"""
        self.groups[group_name].append((worker_id, worker_ref))
        print(f"✅ {worker_id} subscribed to '{group_name}'")
        return f"{worker_id} subscribed to {group_name}"

    def publish(self, sender_id: str, group_name: str, **kwargs):
        """Gửi tin đến tất cả worker trong group (trừ người gửi)"""
        receivers = 0
        if group_name not in self.groups:
            print(f"⚠️ Group '{group_name}' chưa tồn tại")
            return 0
        for wid, ref in self.groups[group_name]:
            if wid != sender_id:
                ref.on_message.remote(sender_id, group_name, **kwargs)
                receivers += 1
        return receivers

    def list_groups(self):
        """Trả về danh sách group và worker hiện có"""
        return {g: [wid for wid, _ in members] for g, members in self.groups.items()}


# ==========================================================
# INTERFACE CLASS: Worker (ABC)
# ==========================================================
class Worker(ABC):
    def __init__(self, worker_id: str, bus):
        self.worker_id = worker_id
        self.bus = bus
        self.inbox = []

    def join_group(self, group_name: str, self_ref):
        """Đăng ký bản thân vào 1 group"""
        return ray.get(self.bus.subscribe.remote(group_name, self_ref, self.worker_id))

    def send_to_group(self, group_name: str, **kwargs):
        """Gửi message tới group"""
        receivers = ray.get(self.bus.publish.remote(self.worker_id, group_name, **kwargs))
        print(f"📤 {self.worker_id} -> '{group_name}' ({receivers} receivers): {kwargs}")
        return receivers

    @abstractmethod
    def on_message(self, sender_id: str, group_name: str, **kwargs):
        """Interface: bắt buộc các subclass phải triển khai"""
        pass


# ==========================================================
# CONCRETE WORKERS: Các lớp triển khai interface Worker
# ==========================================================

@ray.remote
class AlphaWorker(Worker):
    """Worker của group 'alpha'"""
    def on_message(self, sender_id, group_name, **kwargs):
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] 🧩 Alpha[{self.worker_id}] <- {sender_id}@{group_name}: {kwargs}")


@ray.remote
class BetaWorker(Worker):
    """Worker của group 'beta'"""
    def on_message(self, sender_id, group_name, **kwargs):
        msg = kwargs.get("text", "")
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] 🔧 Beta[{self.worker_id}] <- {sender_id}@{group_name}: text='{msg}'")


@ray.remote
class GammaWorker(Worker):
    """Worker của group 'gamma'"""
    def on_message(self, sender_id, group_name, **kwargs):
        payload = kwargs.get("payload", [])
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] 📊 Gamma[{self.worker_id}] <- {sender_id}@{group_name}: data={payload}")


@ray.remote
class DualWorker(Worker):
    """Worker thuộc cả 'alpha' và 'beta'"""
    def on_message(self, sender_id, group_name, **kwargs):
        ts = time.strftime("%H:%M:%S")
        if group_name == "alpha":
            print(f"[{ts}] 🔁 Dual[{self.worker_id}] xử lý alpha: {kwargs}")
        elif group_name == "beta":
            print(f"[{ts}] 🔁 Dual[{self.worker_id}] xử lý beta: {kwargs}")
        else:
            print(f"[{ts}] ⚙️ Dual[{self.worker_id}] không có handler cho {group_name}")


@ray.remote
class UniversalWorker(Worker):
    """Worker có thể join nhiều group (universal handler)"""
    def on_message(self, sender_id, group_name, **kwargs):
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] 🌐 Universal[{self.worker_id}] nhận message từ {sender_id}@{group_name}: {kwargs}")


# ==========================================================
# DEMO CHÍNH
# ==========================================================
if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)

    bus = EventBus.remote()

    # Tạo 5 worker khác nhau (mỗi loại kế thừa interface Worker)
    A = AlphaWorker.remote("A", bus)
    B = BetaWorker.remote("B", bus)
    C = GammaWorker.remote("C", bus)
    D = DualWorker.remote("D", bus)
    E = UniversalWorker.remote("E", bus)

    # Đăng ký vào group
    ray.get([
        A.join_group.remote("alpha", A),
        B.join_group.remote("beta", B),
        C.join_group.remote("gamma", C),
        D.join_group.remote("alpha", D),
        D.join_group.remote("beta", D),
        E.join_group.remote("alpha", E),
        E.join_group.remote("beta", E),
        E.join_group.remote("gamma", E),
    ])

    print("\n📋 Group membership:")
    print(ray.get(bus.list_groups.remote()))

    # Gửi message
    A.send_to_group.remote("alpha", msg="Hello from Alpha A")
    B.send_to_group.remote("beta", text="Training progress = 95%")
    C.send_to_group.remote("gamma", payload=[10, 20, 30])
    D.send_to_group.remote("alpha", update="Sync done")
    D.send_to_group.remote("beta", info="Loss stable")
    E.send_to_group.remote("gamma", note="Broadcast test")

    time.sleep(1)
    ray.shutdown()

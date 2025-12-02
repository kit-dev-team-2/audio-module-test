import sys
import os
import subprocess
import threading
import tkinter as tk
import ast
from tkinter import filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
import json


class ScriptManagerGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Node WS + Python Audio Detector Runner")

        # 프로세스 핸들
        self.node_proc: subprocess.Popen | None = None
        self.py_proc: subprocess.Popen | None = None

        # NEW: WebSocket 클라이언트 목록 관리용
        self.clients: set[str] = set()

        # ================================
        #  Node 서버 영역
        # ================================
        node_frame = tk.LabelFrame(root, text="Node WS 서버 (server.js)")
        node_frame.pack(fill="x", padx=10, pady=5)

        # server.js 경로
        tk.Label(node_frame, text="server.js 경로").grid(
            row=0, column=0, sticky="w", padx=5, pady=2
        )
        self.node_path_var = tk.StringVar(
            value="C:/Users/juuip/open/server3.js"  # 필요 시 수정
        )
        self.node_path_entry = tk.Entry(
            node_frame, textvariable=self.node_path_var, width=50
        )
        self.node_path_entry.grid(row=0, column=1, sticky="we", padx=5, pady=2)
        tk.Button(node_frame, text="찾기...", command=self.browse_node).grid(
            row=0, column=2, padx=5, pady=2
        )

        # HOST, PORT 입력
        tk.Label(node_frame, text="HOST").grid(
            row=1, column=0, sticky="w", padx=5, pady=2
        )
        self.node_host_var = tk.StringVar(value="0.0.0.0")
        tk.Entry(node_frame, textvariable=self.node_host_var, width=15).grid(
            row=1, column=1, sticky="w", padx=5, pady=2
        )

        tk.Label(node_frame, text="PORT").grid(
            row=1, column=2, sticky="e", padx=5, pady=2
        )
        self.node_port_var = tk.StringVar(value="8080")
        tk.Entry(node_frame, textvariable=self.node_port_var, width=8).grid(
            row=1, column=3, sticky="w", padx=5, pady=2
        )

        # 실행/종료 버튼
        tk.Button(node_frame, text="서버 실행", command=self.start_node).grid(
            row=2, column=1, sticky="e", padx=5, pady=5
        )
        tk.Button(node_frame, text="서버 종료", command=self.stop_node).grid(
            row=2, column=2, sticky="w", padx=5, pady=5
        )

        self.clients: set[str] = set()
        self.client_ping: dict[str, int | float] = {}   # IP별 ping ms 저장

        for c in range(4):
            node_frame.columnconfigure(c, weight=1)

        # ================================
        #  Python 감지 영역
        # ================================
        py_frame = tk.LabelFrame(root, text="Python PANNs 감지 스크립트")
        py_frame.pack(fill="x", padx=10, pady=5)

        # 파이썬 스크립트 경로
        tk.Label(py_frame, text="감지 스크립트 경로").grid(
            row=0, column=0, sticky="w", padx=5, pady=2
        )
        self.py_path_var = tk.StringVar(
            value="C:\\Users\\juuip\\open\\sound3.py"  # 파일명에 맞게 수정
        )
        self.py_path_entry = tk.Entry(
            py_frame, textvariable=self.py_path_var, width=50
        )
        self.py_path_entry.grid(row=0, column=1, sticky="we", padx=5, pady=2)
        tk.Button(py_frame, text="찾기...", command=self.browse_python).grid(
            row=0, column=2, padx=5, pady=2
        )

        # BASE_CKPT_PATH
        tk.Label(py_frame, text="BASE_CKPT_PATH").grid(
            row=1, column=0, sticky="w", padx=5, pady=2
        )
        self.base_ckpt_var = tk.StringVar(
            value=r"C:\\Users\\juuip\\open\\Cnn14_mAP=0.431.pth"
        )
        tk.Entry(py_frame, textvariable=self.base_ckpt_var, width=50).grid(
            row=1, column=1, sticky="we", padx=5, pady=2
        )
        tk.Button(py_frame, text="찾기...", command=self.browse_base_ckpt).grid(
            row=1, column=2, padx=5, pady=2
        )

        # FINETUNED_CKPT
        tk.Label(py_frame, text="FINETUNED_CKPT").grid(
            row=2, column=0, sticky="w", padx=5, pady=2
        )
        self.ft_ckpt_var = tk.StringVar(
            value=r"C:\\Users\\juuip\\open\\best_panns6_acc0.828.pt"
        )
        tk.Entry(py_frame, textvariable=self.ft_ckpt_var, width=50).grid(
            row=2, column=1, sticky="we", padx=5, pady=2
        )
        tk.Button(py_frame, text="찾기...", command=self.browse_ft_ckpt).grid(
            row=2, column=2, padx=5, pady=2
        )

        # MIC_SAMPLE_RATE, DEVICE_INDEX
        tk.Label(py_frame, text="MIC_SAMPLE_RATE").grid(
            row=3, column=0, sticky="w", padx=5, pady=2
        )
        self.mic_sr_var = tk.StringVar(value="16000")
        tk.Entry(py_frame, textvariable=self.mic_sr_var, width=10).grid(
            row=3, column=1, sticky="w", padx=5, pady=2
        )

        tk.Label(py_frame, text="DEVICE_INDEX").grid(
            row=3, column=2, sticky="e", padx=5, pady=2
        )
        self.device_idx_var = tk.StringVar(value="1")
        tk.Entry(py_frame, textvariable=self.device_idx_var, width=5).grid(
            row=3, column=3, sticky="w", padx=5, pady=2
        )

        # PRE_BUFFER_DURATION, DETECT_DURATION
        tk.Label(py_frame, text="PRE_BUFFER_DURATION (초)").grid(
            row=4, column=0, sticky="w", padx=5, pady=2
        )
        self.pre_buf_var = tk.StringVar(value="0.2")
        tk.Entry(py_frame, textvariable=self.pre_buf_var, width=10).grid(
            row=4, column=1, sticky="w", padx=5, pady=2
        )

        tk.Label(py_frame, text="DETECT_DURATION (초)").grid(
            row=4, column=2, sticky="e", padx=5, pady=2
        )
        self.detect_buf_var = tk.StringVar(value="0.5")
        tk.Entry(py_frame, textvariable=self.detect_buf_var, width=10).grid(
            row=4, column=3, sticky="w", padx=5, pady=2
        )

        # CONF_THRESH, NODE_WS_URL
        tk.Label(py_frame, text="CONF_THRESH").grid(
            row=5, column=0, sticky="w", padx=5, pady=2
        )
        self.conf_thresh_var = tk.StringVar(value="0.5")
        tk.Entry(py_frame, textvariable=self.conf_thresh_var, width=10).grid(
            row=5, column=1, sticky="w", padx=5, pady=2
        )

        tk.Label(py_frame, text="NODE_WS_URL").grid(
            row=5, column=2, sticky="e", padx=5, pady=2
        )
        self.ws_url_var = tk.StringVar(value="ws://localhost:8080")
        tk.Entry(py_frame, textvariable=self.ws_url_var, width=25).grid(
            row=5, column=3, sticky="w", padx=5, pady=2
        )

        # 실행/종료 버튼
        tk.Button(py_frame, text="감지 시작", command=self.start_python).grid(
            row=6, column=1, sticky="e", padx=5, pady=5
        )
        tk.Button(py_frame, text="감지 종료", command=self.stop_python).grid(
            row=6, column=2, sticky="w", padx=5, pady=5
        )

        for c in range(4):
            py_frame.columnconfigure(c, weight=1)

                # ================================
        #  로그 영역: Node / Python + 클라이언트 목록
        # ================================
        log_frame = tk.Frame(root)
        log_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # 그리드 비율 설정: Node:Python = 2:1, 클라이언트는 고정 느낌
        log_frame.columnconfigure(0, weight=3)   # Node 로그 넓게
        log_frame.columnconfigure(1, weight=1)   # Python 로그 조금 좁게
        log_frame.columnconfigure(2, weight=0)   # 클라이언트 리스트 (필요한 만큼만)
        log_frame.rowconfigure(0, weight=1)  

               # Node 로그
        node_log_frame = tk.LabelFrame(log_frame, text="Node 로그")
        node_log_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        self.node_log_box = ScrolledText(node_log_frame, height=20, width=70)  # ★ width 추가
        self.node_log_box.pack(fill="both", expand=True)

        tk.Button(
            node_log_frame,
            text="Node 로그 Clear",
            command=self.clear_node_log,
        ).pack(fill="x", padx=5, pady=3)

        # Python 로그
        py_log_frame = tk.LabelFrame(log_frame, text="Python 로그")
        py_log_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 5))
        self.py_log_box = ScrolledText(py_log_frame, height=20, width=40)  # ★ 더 좁게
        self.py_log_box.pack(fill="both", expand=True)

        tk.Button(
            py_log_frame,
            text="Python 로그 Clear",
            command=self.clear_python_log,
        ).pack(fill="x", padx=5, pady=3)

        # 연결된 클라이언트 목록 (오른쪽)
        client_frame = tk.LabelFrame(log_frame, text="연결된 WS 클라이언트")
        client_frame.grid(row=0, column=2, sticky="ns", padx=(5, 0))
        client_frame.config(width=180)  # 적당한 기본 폭

        self.client_listbox = tk.Listbox(client_frame, height=20)
        self.client_listbox.pack(fill="both", expand=True, padx=5, pady=5)

        # 창 닫을 때 정리
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ================================
    #  파일 선택
    # ================================
    def browse_node(self):
        path = filedialog.askopenfilename(
            filetypes=[("JavaScript", "*.js"), ("All files", "*.*")]
        )
        if path:
            self.node_path_var.set(path)

    def browse_python(self):
        path = filedialog.askopenfilename(
            filetypes=[("Python", "*.py"), ("All files", "*.*")]
        )
        if path:
            self.py_path_var.set(path)

    def browse_base_ckpt(self):
        path = filedialog.askopenfilename(
            filetypes=[("PyTorch Checkpoint", "*.pth;*.pt"), ("All files", "*.*")]
        )
        if path:
            self.base_ckpt_var.set(path)

    def browse_ft_ckpt(self):
        path = filedialog.askopenfilename(
            filetypes=[("PyTorch Checkpoint", "*.pth;*.pt"), ("All files", "*.*")]
        )
        if path:
            self.ft_ckpt_var.set(path)

    # ================================
    #  Node 서버 실행/종료
    # ================================
    def start_node(self):
        if self.node_proc is not None and self.node_proc.poll() is None:
            messagebox.showinfo("알림", "Node 서버가 이미 실행 중입니다.")
            return

        path = self.node_path_var.get().strip()
        if not path:
            messagebox.showwarning("경고", "server.js 경로를 입력하세요.")
            return

        host = self.node_host_var.get().strip() or "0.0.0.0"
        port = self.node_port_var.get().strip() or "8080"

        env = os.environ.copy()
        env["HOST"] = host
        env["PORT"] = port

        # NEW: Node 재시작 시 클라이언트 목록 초기화
        self.clients.clear()
        self._refresh_client_listbox()

        try:
            cmd = ["node", path]
            self._append_log(
                self.node_log_box,
                f"[NODE] 실행: {' '.join(cmd)} (HOST={host}, PORT={port})\n",
            )
            self.node_proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )
            threading.Thread(
                target=self._reader_thread,
                args=(self.node_proc, self.node_log_box, "[NODE]"),
                daemon=True,
            ).start()
        except FileNotFoundError:
            messagebox.showerror(
                "에러",
                "node 명령어를 찾을 수 없습니다. Node.js가 설치되어 있는지 확인하세요.",
            )
        except Exception as e:
            messagebox.showerror("에러", f"Node 서버 실행 중 예외 발생:\n{e}")

    def stop_node(self):
        if self.node_proc is not None and self.node_proc.poll() is None:
            self.node_proc.terminate()
            self._append_log(self.node_log_box, "[NODE] 서버 종료 요청\n")
        else:
            self._append_log(self.node_log_box, "[NODE] 실행 중인 서버가 없습니다.\n")

        # NEW: 서버 종료 시 클라이언트 목록도 리셋
        self.clients.clear()
        self._refresh_client_listbox()
    
    def _handle_python_special_log(self, line: str):
        line = line.strip()
        key = "[CONFIG] updated:"
        if key not in line:
            return

        try:
            cfg_str = line.split(key, 1)[1].strip()
            cfg = ast.literal_eval(cfg_str)   # {'CONF_THRESH': 0.7, ...}
        except Exception:
            return

        def _update():
            if "CONF_THRESH" in cfg:
                self.conf_thresh_var.set(str(cfg["CONF_THRESH"]))
            if "DETECT_DURATION" in cfg:
                self.detect_buf_var.set(str(cfg["DETECT_DURATION"]))
            if "PRE_BUFFER_DURATION" in cfg:
                self.pre_buf_var.set(str(cfg["PRE_BUFFER_DURATION"]))

        self.root.after(0, _update)

    # ================================
    #  Python 스크립트 실행/종료
    # ================================
    def start_python(self):
        if self.py_proc is not None and self.py_proc.poll() is None:
            messagebox.showinfo("알림", "Python 감지 스크립트가 이미 실행 중입니다.")
            return

        path = self.py_path_var.get().strip()
        if not path:
            messagebox.showwarning("경고", "파이썬 스크립트 경로를 입력하세요.")
            return

        env = os.environ.copy()
        env["MIC_SAMPLE_RATE"] = self.mic_sr_var.get().strip() or "16000"
        env["DEVICE_INDEX"] = self.device_idx_var.get().strip() or "1"
        env["CONF_THRESH"] = self.conf_thresh_var.get().strip() or "0.5"
        env["NODE_WS_URL"] = self.ws_url_var.get().strip() or "ws://localhost:8080"
        env["PRE_BUFFER_DURATION"] = self.pre_buf_var.get().strip() or "0.2"
        env["DETECT_DURATION"] = self.detect_buf_var.get().strip() or "0.5"
        env["BASE_CKPT_PATH"] = self.base_ckpt_var.get().strip()
        env["FINETUNED_CKPT"] = self.ft_ckpt_var.get().strip()

        try:
            cmd = [sys.executable, path]
            self._append_log(
                self.py_log_box,
                (
                    f"[PY] 실행: {' '.join(cmd)}\n"
                    f"     MIC={env['MIC_SAMPLE_RATE']} DEV={env['DEVICE_INDEX']}\n"
                    f"     PRE={env['PRE_BUFFER_DURATION']} DETECT={env['DETECT_DURATION']}\n"
                    f"     CONF={env['CONF_THRESH']} WS={env['NODE_WS_URL']}\n"
                    f"     BASE_CKPT={env['BASE_CKPT_PATH']}\n"
                    f"     FT_CKPT={env['FINETUNED_CKPT']}\n"
                ),
            )
            self.py_proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )
            threading.Thread(
                target=self._reader_thread,
                args=(self.py_proc, self.py_log_box, "[PY]"),
                daemon=True,
            ).start()
        except Exception as e:
            messagebox.showerror("에러", f"Python 스크립트 실행 중 예외 발생:\n{e}")

    def stop_python(self):
        if self.py_proc is not None and self.py_proc.poll() is None:
            self.py_proc.terminate()
            self._append_log(self.py_log_box, "[PY] 감지 스크립트 종료 요청\n")
        else:
            self._append_log(self.py_log_box, "[PY] 실행 중인 감지 스크립트가 없습니다.\n")

    # ================================
    #  공통: 프로세스 출력 읽기
    # ================================
    def _reader_thread(self, proc: subprocess.Popen, widget: ScrolledText, prefix: str):
        try:
            for line in iter(proc.stdout.readline, ""):
                if not line:
                    break

                # Node 쪽 [PING] 로그는 텍스트 창에는 안 찍고,
                #    클라이언트 RTT 갱신만 하고 넘어감
                if prefix == "[NODE]" and line.strip().startswith("[PING]"):
                    self._update_clients_from_node_log(line)
                    continue

                # 나머지 라인들은 그대로 출력
                self._append_log(widget, f"{prefix} {line}")

                if prefix == "[NODE]":
                    self._update_clients_from_node_log(line)
                elif prefix == "[PY]":
                    self._handle_python_special_log(line)
        finally:
            code = proc.poll()
            self._append_log(widget, f"{prefix} 프로세스 종료 (code={code})\n")

    def _append_log(self, widget: ScrolledText, text: str):
        def _append():
            widget.insert(tk.END, text)
            widget.see(tk.END)

        self.root.after(0, _append)

    # NEW: Node 로그 Clear
    def clear_node_log(self):
        self.node_log_box.delete("1.0", tk.END)

    # NEW: Python 로그 Clear
    def clear_python_log(self):
        self.py_log_box.delete("1.0", tk.END)

    # NEW: Node 로그 한 줄에서 클라이언트 접속/종료 파싱
    def _update_clients_from_node_log(self, line: str):
        line = line.strip()
        action = None
        ip = None

        # 1) 설정 변경 로그: [CONFIG] { ...json... }
        if line.startswith("[CONFIG]"):
            try:
                json_str = line.split("[CONFIG]", 1)[1].strip()
                payload = json.loads(json_str)
                cfg = payload.get("config", {})
            except Exception:
                return

            def _update_cfg():
                if "CONF_THRESH" in cfg:
                    self.conf_thresh_var.set(str(cfg["CONF_THRESH"]))
                if "DETECT_DURATION" in cfg:
                    self.detect_buf_var.set(str(cfg["DETECT_DURATION"]))
                if "PRE_BUFFER_DURATION" in cfg:
                    self.pre_buf_var.set(str(cfg["PRE_BUFFER_DURATION"]))

            self.root.after(0, _update_cfg)
            return

        # 접속/종료
        if "WS connected:" in line:
            parts = line.split("WS connected:")
            if len(parts) >= 2:
                ip = parts[1].strip()
                action = "add"
        elif "WS closed" in line:
            parts = line.split("WS closed")
            if len(parts) >= 2:
                ip = parts[1].strip()
                action = "remove"

        # ping RTT 로그: "[PING] 127.0.0.1 rtt=37ms"
        if line.startswith("[PING]"):
            try:
                tokens = line.split()
                # [0]="[PING]", [1]=ip, [2]="rtt=37ms"
                ip = tokens[1]
                rtt_token = tokens[2]          # "rtt=37ms"
                rtt_str = rtt_token.split("=")[1].rstrip("ms")
                rtt = float(rtt_str)
            except Exception:
                return

            def _update_ping():
                self.clients.add(ip)           # 혹시 없으면 추가
                self.client_ping[ip] = rtt
                self._refresh_client_listbox()

            self.root.after(0, _update_ping)
            return

        if not action or not ip:
            return

        def _update():
            if action == "add":
                self.clients.add(ip)
            elif action == "remove":
                self.clients.discard(ip)
                self.client_ping.pop(ip, None)
            self._refresh_client_listbox()

        self.root.after(0, _update)


    # NEW: Listbox 갱신
    def _refresh_client_listbox(self):
        self.client_listbox.delete(0, tk.END)
        for addr in sorted(self.clients):
            rtt = self.client_ping.get(addr)
            if rtt is not None:
                label = f"{addr} ({int(rtt)} ms)"
            else:
                label = addr
            self.client_listbox.insert(tk.END, label)


    # ================================
    #  종료 처리
    # ================================
    def on_close(self):
        try:
            if self.node_proc is not None and self.node_proc.poll() is None:
                self.node_proc.terminate()
            if self.py_proc is not None and self.py_proc.poll() is None:
                self.py_proc.terminate()
        finally:
            self.root.destroy()


def main():
    root = tk.Tk()
    app = ScriptManagerGUI(root)
    root.geometry("1100x700")
    root.mainloop()


if __name__ == "__main__":
    main()

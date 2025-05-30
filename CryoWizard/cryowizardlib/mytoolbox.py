import threading
import socket
import psutil
import pickle
import queue
import json
import yaml
import os



class ThreadPool():
    # 线程池
    def __init__(self, maxsize = 8):
        self.pool = queue.Queue(maxsize)   # 使用queue队列，创建一个线程池
        for _ in range(maxsize):
            self.pool.put(threading.Thread)

    def get_thread(self):
        return self.pool.get()

    def add_thread(self):
        self.pool.put(threading.Thread)


class MultiThreadingRun():
    # 多线程运行自定义函数
    def __init__(self, threadpoolmaxsize = 8):
        self.threadpool = ThreadPool(threadpoolmaxsize)
        self.global_thread_lock = threading.Lock()

    def runwiththreadlock(self, function, **kwargs):
        # 用法举例：
        # def testdef(a, b, c):
        #     print('Sum:', a + b + c)
        # multithread = mytoolbox.MultiThreadingRun(2)
        # multithread.runwiththreadlock(testdef, a=1, b=2, c=3)
        with self.global_thread_lock:
            function(**kwargs)

    def setthread(self, function, **kwargs):
        # 用法举例：
        # def testdef(a, b, c):
        #     print('Sum:', a + b + c)
        # multithread = mytoolbox.MultiThreadingRun(2)
        # multithread.setthread(testdef, a=1, b=2, c=3)
        def tempfunction():
            function(**kwargs)
            self.threadpool.add_thread()
        readythread = self.threadpool.get_thread()
        process = readythread(target=tempfunction)
        process.start()

    def ifthreadpoolfull(self):
        return self.threadpool.pool.full()


def killprocesswithpid(pid):
    # 通过进程pid杀掉进程，如果有子进程也会递归杀掉子进程后再杀掉进程
    # pid为int类型
    target_process = psutil.Process(pid)
    for child_process in target_process.children():
        killprocesswithpid(child_process.pid)
    target_process.kill()

def scanusableport(port_list):
    # 从port_list中查找一个能用的port并返回
    def checkport(port):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(('localhost', port))
            s.close()
            return True
        except:
            return False
    for port in port_list:
        if checkport(port):
            return port
    return None

def savetojson(savingdata, filename, ifprint=True):
    # 将中间数据存储为json，适合列表、字典、字符串等
    # 文件若已存在会自动覆盖
    if os.path.exists(filename):
        os.remove(filename)
    with open(filename, 'w') as f:
        json.dump(savingdata, f, indent=4)
    if ifprint:
        print('Saving', filename, 'done!', flush=True)

def readjson(filename):
    # 读取存储在json中的数据
    with open(filename, 'r') as f:
        savingdata = json.load(f)
    return savingdata

def savebypickle(savingdata, filename, ifprint=True):
    # 将中间数据使用pickle存储为二进制文件
    # 文件若已存在会自动覆盖
    if os.path.exists(filename):
        os.remove(filename)
    with open(filename, 'wb') as f:
        pickle.dump(savingdata, f)
    if ifprint:
        print('Saving', filename, 'done!', flush=True)

def readpicklefile(filename):
    # 读取使用pickle存储的文件
    with open(filename, 'rb') as f:
        savingdata = pickle.load(f)
    return savingdata

def savetoyaml(savingdata, filename, ifprint=True):
    # 将中间数据存储为yaml
    # 文件若已存在会自动覆盖
    if os.path.exists(filename):
        os.remove(filename)
    with open(filename, 'w') as f:
        yaml.dump(data=savingdata, stream=f, allow_unicode=True)
    if ifprint:
        print('Saving', filename, 'done!', flush=True)

def readyaml(filename):
    # 读取yaml文件
    with open(filename, 'r') as f:
        savingdata = yaml.load(f.read(), Loader=yaml.FullLoader)
    return savingdata
import threading
import time

class StoppableThread(threading.Thread):
    ''' Stoppable Thread using threading.Event() '''

    def __init__(self, repeat_fn, args=(), kwargs={}, name=''):
        super().__init__()
        self._stopevent = threading.Event()
        self.name = name
        self.fn = repeat_fn
        self.args = args
        self.kwargs = kwargs

    def run(self):
        while not self._stopevent.is_set():
            self.fn(*self.args, **self.kwargs)

    def stop(self, timeout=None):
        self._stopevent.set()
        super().join(timeout)
        # print('thread', self.name, 'terminated')



class PeriodicTimer(threading.Thread):
    ''' Periodic Timer '''

    def __init__(self, period, repeat_fn, args=(), kwargs={}, name=''):
        super().__init__()
        self._stopevent = threading.Event()
        self.name = name
        self.fn = repeat_fn
        self.args = args
        self.kwargs = kwargs

        self.period = period

    def run(self):
        # while not self._stopevent.wait(0.015):
        delay = 0.001
        while not self._stopevent.wait(delay):
            t0 = time.time()
            self.fn(*self.args, **self.kwargs)
            # self._stopevent.wait(self.period)
            dt = time.time()-t0
            delay = self.period-dt if dt<self.period else 0.001
            # time.sleep(self.period-dt)
            # self._stopevent.wait(self.period-dt)

    def stop(self, timeout=None):
        self._stopevent.set()

## class PeriodicTimer:
## 
##     def __init__(self, period, repeat_fn, args=(), kwargs={}):
##         self.dt = period
##         self.fn = repeat_fn
##         self.args = args
##         self.kwargs = kwargs
## 
##         self.timer = threading.Timer(self.dt, self.run)
## 
##     def start(self):
##         if not self.timer.is_alive():
##             self.timer.start()
## 
##     def run(self):
## 
##         self.fn(*self.args,**self.kwargs)
##         # print(self.timer)
## 
##         self.timer = threading.Timer(self.dt, self.run)
##         self.timer.start()
## 
##     def stop(self):
##         if self.timer.is_alive():
##             self.timer.cancel()
## 
##     def is_alive(self):
##         return self.timer.is_alive()


if __name__ == '__main__':

    import time

    t0 = time.time()
    def worker(wait=True):
        print(' - worker function: %.2f'%(time.time()-t0,))
        if wait:
            time.sleep(0.2)


    def test_stoppable_thread():

        """ Stoppable Thread Test """

        th = StoppableThread(worker)
        th.start()

        while True:
            try:
                time.sleep(10)
            except KeyboardInterrupt:
                th.stop()
                break;
        #
        print()
        print('thread alive?', th.is_alive())
        print('main thread waits for 1 sec')
        time.sleep(1)


    def test_periodic_timer():
        """ Periodic Timer """

        timer = PeriodicTimer(0.5, worker, args=(False,))
        timer.start()

        while True:
            try:
                time.sleep(10)
            except KeyboardInterrupt:
                timer.stop()
                break;
        #
        print()
        print('timer alive?', timer.is_alive())
        print('main thread waits for 1 sec')
        time.sleep(1)



    # test 1
    test_stoppable_thread()

    # test 2
    test_periodic_timer()




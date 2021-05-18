from multiprocessing import Process, Queue

def worker(name, evalfunc, querries, answers, model):
    while True:
        querry = querries.get()
        answer = evalfunc(model, querry)
        answers.put(answer)


    
class Pool:
    """ Pool of workers. Workers consume tasks from the querries queue and 
        feed answers to the answers queue.
    """

    
    def __init__(self, processors, evalfunc, models):

        self.querries = Queue()
        self.answers = Queue()

        self.workers = []
        for i in range(processors):
             worker_i = Process(target=worker, args=(i, evalfunc, self.querries, self.answers, models[i]))
             self.workers.append(worker_i)
             worker_i.start()        # Launch worker() as a separate python process

    def putQuerry(self, querry):
        self.querries.put(querry)

    def getAnswer(self):
        return self.answers.get()
             

    def close(self):
        for w in self.workers:
            w.terminate()
        #print("pool killed")
        

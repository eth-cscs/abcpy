from mpi4py import MPI
import sys

__mpimanager = None

class MPIManager(object):
    """Defines the behavior of the slaves/worker processes

    This class construct the MPI communicators structure needed
    if the rank of the process is in scheduler_node_ranks, the process is a scheduler
    then there is process_per_model process per communicator
    """

    def __init__(self, scheduler_node_ranks=[0], process_per_model=1):
        """
        Parameters
        ----------
        scheduler_node_ranks: Python list
            list of ranks computation should not happen on.
            Should include the scheduler so it doesn't get 
            overwhelmed with work.

        process_per_model: Integer
            the number of process to allow to each model
       """

        self._world_communicator = MPI.COMM_WORLD
        self._size = self._world_communicator.Get_size()
        self._rank = self._world_communicator.Get_rank()

        #Construct the appropriate communicators for resource allocation to models
        #There is one communicator for scheduler nodes
        #And one communicator per model
        self._scheduler_node_ranks = scheduler_node_ranks
        self._process_per_model = process_per_model
        self._model_color = int(((self._rank - sum(i < self._rank for i in scheduler_node_ranks)) / process_per_model) + 1)
        if(self._rank in scheduler_node_ranks):
            self._model_color = 0
        self._model_communicator = MPI.COMM_WORLD.Split(self._model_color, self._rank)
        self._model_size = self._model_communicator.Get_size()
        self._model_rank = self._model_communicator.Get_rank()

        # create a communicator to broadcast instructions to slaves
        self._scheduler_color = 1
        if(self._model_color == 0 or self._model_rank == 0):
            self._scheduler_color = 0
        self._scheduler_communicator = MPI.COMM_WORLD.Split(self._scheduler_color, self._rank)
        self._scheduler_size = self._scheduler_communicator.Get_size()
        self._scheduler_rank = self._scheduler_communicator.Get_rank()

        self._leader = False
        self._scheduler = False
        self._team = False
        self._worker = False

        if self._rank == 0:
            self._scheduler = True
        elif self._model_rank == 0:
            self._team = True
            self._leader = True
        else:
            self._team = True
            self._worker = True


    def is_scheduler(self):
        ''' Tells if the process is a scheduler '''
        return self._scheduler

    def is_team(self):
        ''' Tells if the process is a team '''
        return self._team

    def is_leader(self):
        ''' Tells if the process is a leader '''
        return self._leader

    def is_worker(self):
        ''' Tells if the process is a worker '''
        return self._worker

    def get_scheduler_node_ranks(self):
        ''' Returns the list of scheduler node wanks '''
        return self._scheduler_node_ranks

    def get_world_rank(self):
        ''' Returns the current rank '''
        return self._rank

    def get_world_size(self):
        ''' Returns the size of the world communicator '''
        return self._size

    def get_world_communicator(self):
        ''' Returns the world communicator '''
        return self._world_communicator

    def get_model_rank(self):
        ''' Returns the rank in the world communicator '''
        return self._model_rank

    def get_model_size(self):
        ''' Returns the size of the model communicator '''
        return self._model_size

    def get_model_communicator(self):
        ''' Returns the model communicator '''
        return self._model_communicator

    def get_scheduler_rank(self):
        ''' Returns the rank in the scheduler communicator '''
        return self._scheduler_rank

    def get_scheduler_size(self):
        ''' Returns the size of the scheduler communicator '''
        return self._scheduler_size

    def get_scheduler_communicator(self):
        ''' Returns the scheduler communicator '''
        return self._scheduler_communicator

def get_mpi_manager():
    ''' Return the instance of mpimanager
    Creates one with default parameters is not already existing '''
    global mpimanager
    if mpimanager == None :
        create_mpi_manager([0], 1)
    return mpimanager

def create_mpi_manager(scheduler_node_ranks, process_per_model):
    ''' Creates the instance of mpimanager with given parameters '''
    global mpimanager
    mpimanager = MPIManager(scheduler_node_ranks, process_per_model)
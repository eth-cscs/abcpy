from mpi4py import MPI
import sys

mpimanager = None

class MPIManager(object):

    def __init__(self, master_node_ranks=[0], process_per_model=1):
        self.world_communicator = MPI.COMM_WORLD
        self.size = self.world_communicator.Get_size()
        self.rank = self.world_communicator.Get_rank()

        #Construct the appropriate communicators for resource allocation to models
        #There is one communicator for master nodes
        #And one communicator per model
        self.master_node_ranks = master_node_ranks
        self.process_per_model = process_per_model
        self.model_color = int(((self.rank - sum(i < self.rank for i in master_node_ranks)) / process_per_model) + 1)
        if(self.rank in master_node_ranks):
            self.model_color = 0
        self.model_communicator = MPI.COMM_WORLD.Split(self.model_color, self.rank)
        self.model_size = self.model_communicator.Get_size()
        self.model_rank = self.model_communicator.Get_rank()

        # create a communicator to broadcast instructions to slaves
        self.master_color = 1
        if(self.model_color == 0 or self.model_rank == 0):
            self.master_color = 0
        self.master_communicator = MPI.COMM_WORLD.Split(self.master_color, self.rank)
        self.master_size = self.master_communicator.Get_size()
        self.master_rank = self.master_communicator.Get_rank()

        self.leader = False
        self.scheduler = False
        self.team = False
        self.worker = False

        if self.rank == 0:
            self.scheduler = True
        elif self.model_rank == 0:
            self.team = True
            self.leader = True
        else:
            self.team = True
            self.worker = True


    def is_scheduler(self):
        return self.scheduler

    def is_team(self):
        return self.team

    def is_leader(self):
        return self.leader

    def is_worker(self):
        return self.worker

    def get_master_node_ranks(self):
        return self.master_node_ranks

    def get_world_rank(self):
        return self.rank

    def get_world_size(self):
        return self.size

    def get_world_communicator(self):
        return self.world_communicator

    def get_model_rank(self):
        return self.model_rank

    def get_model_size(self):
        return self.model_size

    def get_model_communicator(self):
        return self.model_communicator

    def get_master_rank(self):
        return self.master_rank

    def get_master_size(self):
        return self.master_size

    def get_master_communicator(self):
        return self.master_communicator

def get_mpi_manager():
    global mpimanager
    # Error prone ?
    if mpimanager == None :
        create_mpi_manager([0], 1)
    return mpimanager

def create_mpi_manager(master_node_ranks, process_per_model):
    global mpimanager
    mpimanager = MPIManager(master_node_ranks, process_per_model)
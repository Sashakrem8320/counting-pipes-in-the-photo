from pipe_search import rectangular_pipes
from pipe_search import round_pipes
def start(path, nr, iden):
    if nr == 1:
        return rectangular_pipes.start_processing(path, iden)
    elif nr == 0:
       return round_pipes.start_processing(path, iden)
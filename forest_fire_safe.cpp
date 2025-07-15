#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cassert>
#include <cmath>

// States: 0 = empty, 1 = living tree, 2 = burning, 3 = dead
void print_grid(const std::vector<int>& grid, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << grid[i * N + j] << " ";
        }
        std::cout << std::endl;
    }
}
// Function to read grid from file. The file should contain:
// Then N lines each with N integers representing cell state (0 or 1)
std::vector<int> read_grid(const std::string &filename, int &N) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Read all numbers from the file first
    std::vector<int> temp_grid;
    int value;
    while (infile >> value) {
        temp_grid.push_back(value);
    }
    infile.close();

    // Determine grid size N (assuming it's a square grid)
    int total_cells = temp_grid.size();
    N = static_cast<int>(std::sqrt(total_cells));
    
    // Verify that it's a perfect square
    if (N * N != total_cells) {
        std::cerr << "Error: Number of elements in grid is not a perfect square" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    return temp_grid;
}

// Function to generate a random grid of size N with tree probability p
// A cell is set to a living tree (state 1) with probability p, else empty (state 0)
std::vector<int> generate_grid(int N, double p, int seed_offset) {
    std::vector<int> grid(N * N, 0);
    srand(time(NULL) + seed_offset);
    for (int i = 0; i < N * N; i++) {
        double r = (double)rand() / RAND_MAX;
        if (r < p)
            grid[i] = 1;  // living tree
        else
            grid[i] = 0;  // empty
    }
    return grid;
}

// Given global grid coordinates, set top row burning
// (i.e. if cell is living (state 1) in row 0, set to burning (state 2))
void ignite_top_row(std::vector<int>& grid, int N) {
    for (int j = 0; j < N; j++) {
        int idx = j; // row0: index = 0*N + j
        if (grid[idx] == 1) {
            grid[idx] = 2;  // burning
        }
    }
}

// divide the image over MPI tasks using slices
void distribute_grid(int N, int iproc, int nproc, int& i0, int& i1){

    i0 = 0;
    i1 = N;
    if (nproc > 1){
      int ni = N / nproc; 
      i0 = iproc * ni;
      i1 = i0 + ni;
      // make sure we take care of the fact that the grid might not easily divide into slices
      if (iproc == nproc - 1){
        i1 = N;
      }
    }
    
    //std::cout << "MPI task " << iproc << " of " << nproc << " is responsible for " << i0 << "<=i<" << i1 << " (" << i1 - i0 << " rows)" << std::endl;
}

// Given a local 2D array stored in a 1D vector 
// index it as: local[i][j] where i=0..local_rows+1, j=0..N-1.
inline int get_1d_index(int i, int j, int N) {
    return i * N + j;
}

// Update the local grid according to the forest fire rules. 
// old_grid: current state (with ghost rows at index 0 and local_rows+1)
// new_grid: new state to be computed for the owned rows (i=1 to local_rows)
// i0 and i1 starting and ending index for the current task
bool update_fire(int N, int i0, int i1, int iproc, int nproc, std::vector < int > & old_grid,
    std::vector < int >& new_grid, double& calc_time, double& comm_time, bool& global_reached_bottom) {
    bool local_burning = false;
    double t1 = MPI_Wtime();
    // Process real rows: i from 1 to local_rows (1-indexed in our local grid with ghost rows)
    for (int i = i0; i < i1; i++) {
        for (int j = 0; j < N; j++) {

            int idx = get_1d_index(i, j, N);
            int state = old_grid[idx];
            // Default: remains same
            int new_state = state;
            if (state == 1) { // living tree; check neighbours for burning
            bool neigh_burning = false;
            // Up (i-1, j)
            if (i > 0 && old_grid[get_1d_index(i - 1, j, N)] == 2)
                neigh_burning = true;
            // Down (i+1, j)
            if (i < N-1 && old_grid[get_1d_index(i + 1, j, N)] == 2)
                neigh_burning = true;
            // Left (i, j-1)
            if (j > 0 && old_grid[get_1d_index(i, j - 1, N)] == 2)
                neigh_burning = true;
            // Right (i, j+1)
            if (j < N-1 && old_grid[get_1d_index(i, j + 1, N)] == 2)
                neigh_burning = true;
            if (neigh_burning)
                new_state = 2;  // becomes burning
            } else if (state == 2) { // burning tree becomes dead
            new_state = 3;
            }
            new_grid[idx] = new_state;
            if (new_state == 2)
            local_burning = true;
            }
        }
        double t2 = MPI_Wtime(); 
        
    if (nproc > 1){

        // send rows in one direction
        // first send from even tasks and receive on odd
        // then send from odd and receive on even
        // to do this we'll create a new odd/even loop     
        for (int oe=0;oe<=1;oe++){

            // send
            if (iproc > 0 && iproc%2 == oe){
            // find the index of the first tree on row i0
            int ind = get_1d_index(i0, 0, N);
            // send the data, where each row has N elements, using i0 as the tag
            //std::cout << "Sending1 row " << i0 << " (ind=" << ind << ") from task " << iproc << " to task " << iproc-1 << std::endl;
            MPI_Send(&new_grid[ind], N, MPI_INT, iproc-1, i0, MPI_COMM_WORLD);
            }
            
            // receive
            if (iproc < nproc - 1 && iproc%2 == (oe+1)%2){
            // find the index of the first tree on row i1 - this is only guaranteed to work if j0=0, so we could put 0 instead (see comment on BB)
            int ind = get_1d_index(i1, 0, N);
            // receive the data, where each row has N elements, using i1 as the tag
            //std::cout << "Receiving1 row " << i1 << " (ind=" << ind << ") on task " << iproc << " from task " << iproc+1 << std::endl;
            MPI_Recv(&old_grid[ind], N, MPI_INT, iproc+1, i1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        
        // now send in the other direction, again using our even/odd split
        for (int oe=0;oe<=1;oe++){
        
            // send
            if (iproc < nproc - 1 && iproc%2 == oe){
            // find the index of the first tree on row i1-1
            int ind = get_1d_index(i1-1, 0, N);
            // send the data, where each row has N elements, using i1-1 as the tag
            //std::cout << "Sending2 row " << i1-1 << " (ind=" << ind << ") from task " << iproc << " to task " << iproc+1 << std::endl;
            MPI_Send(&new_grid[ind], N, MPI_INT, iproc+1, i1-1, MPI_COMM_WORLD);
            }
            
            // receive
            if (iproc > 0 && iproc%2 == (oe+1)%2){
            // find the index of the first tree on row i0-1
            int ind = get_1d_index(i0-1, 0, N);
            // receive the data, where each row has N elements, using i1 as the tag
            //std::cout << "Receiving2 row " << i0-1 << " (ind=" << ind << ") on task " << iproc << " from task " << iproc-1 << std::endl;
            MPI_Recv(&old_grid[ind], N, MPI_INT, iproc-1, i0-1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }     
        
        // copy the elements belonging to this task from new to old
        for (int i=i0;i<i1;i++){
            for (int j=0;j<N;j++){
                int ind = get_1d_index(i, j, N);
                old_grid[ind] = new_grid[ind];
            }
            }
        } 
        else{
            // we only have 1 MPI task, so we can copy as before
            old_grid = new_grid;  
        }

        
        bool local_reached_bottom = false;
        // Check if fire reached bottom (only for last process)
        if (iproc == nproc - 1) {
            for (int j = 0; j < N; j++) {
                if (new_grid[get_1d_index(N-1, j, N)] == 2) {  // Local index of bottom row
                    local_reached_bottom = true;
                    break;
                }
            }
        }

        MPI_Allreduce(&local_reached_bottom, &global_reached_bottom, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);

        bool global_burning = false;
        MPI_Allreduce(&local_burning, &global_burning, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
        // Reduce the reached_bottom status
        
    double t3 = MPI_Wtime(); 
    calc_time += t2 - t1;
    comm_time += t3 - t2;
    
    return global_burning;
}

int main(int argc, char* argv[]) {

      ////////////////////////////////////////////////////////
     //                   Initialise MPI                   //
    ////////////////////////////////////////////////////////  

    // initialise MPI
    MPI_Init(&argc, &argv);

    // Get the number of processes in MPI_COMM_WORLD
    int nproc;
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    // get the rank of this process in MPI_COMM_WORLD
    int iproc;
    MPI_Comm_rank(MPI_COMM_WORLD, &iproc);

    ////////////////////////////////////////////////////////
   //      Read from file or Generate a random grid      //
  //////////////////////////////////////////////////////// 

    double t1 = MPI_Wtime();

    // Command-line arguments:
    // Two modes: "rand" or "file"
    // For "rand": usage: ./forest_fire rand N p M
    //   N: grid size (square grid N x N)
    //   p: probability that a cell contains a living tree initially
    //   M: number of independent runs (each with a different random seed)
    // For "file": usage: ./forest_fire file grid_filename
    if (argc < 2) {
        if (iproc == 0)
            std::cerr << "Usage: For random grid: " << argv[0] << " rand N p M\n"
                      << "       For file input: " << argv[0] << " file grid_filename" << std::endl;
        MPI_Finalize();
        return 1;
    }

    std::string mode = argv[1];
    int N;           // grid dimension
    double p;  // tree probability
    int M;       // number of runs (averaging over independent grids)
    std::vector<int> old_grid; // vector to store the grid
    
    if (mode == "file") {
        if (argc != 3) {
            if (iproc == 0)
                std::cerr << "For file mode, usage: " << argv[0] << " file grid_filename" << std::endl;
            MPI_Finalize();
            return 1;
        }
        if (iproc == 0) {
            old_grid = read_grid(argv[2], N);
        }
        // first share the image size
        MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
        M = 1; // file input only runs once
    }else if (mode == "rand") {
        if (argc != 5) {
            if (iproc == 0)
                std::cerr << "For random grid mode, usage: " << argv[0] << " rand N p M" << std::endl;
            MPI_Finalize();
            return 1;
        }
        N = atoi(argv[2]);
        p = atof(argv[3]);
        M = atoi(argv[4]);
    } else {
        if (iproc == 0)
            std::cerr << "Unknown mode: " << mode << std::endl;
        MPI_Finalize();
        return 1;
    }
    
    //Initialize a new grid
    std::vector < int > new_grid(N*N, 0);
    
    // divide the rows among MPI tasks
    int i0, i1;
    distribute_grid(N, iproc, nproc, i0, i1);

    double total_steps_sum = 0.0;
    double total_reached_bottom = 0.0;
    double total_sim_time = 0.0;

    for (int run = 0; run < M; run++){
        if (mode == "rand"){
            // generate_grid on the root task
            if (iproc == 0){
                old_grid = generate_grid(N, p, run);
            }
        }
        // broadcast the data
        if (nproc > 1){
            // now broadcast the actual grid data
            if (iproc!=0){
                old_grid.resize(N*N, 0);
            }
            MPI_Bcast(old_grid.data(), N*N, MPI_INT, 0, MPI_COMM_WORLD);      
        }

        // Rank 0: ignite the top row (global row 0)
        if (iproc == 0) {
            ignite_top_row(old_grid, N);
        }
        
        // Run the model
        bool burning = true;
        bool sim_reached_bottom = false;
        int steps = 0;
    
        double calc_time = 0;
        double comm_time = 0;
        
        // just to be sure the tasks are still in sync
        MPI_Barrier(MPI_COMM_WORLD);
        double sim_start = MPI_Wtime();
        
        while (burning){
            bool step_reached_bottom = false;
            // Update the grid
            burning = update_fire(N, i0, i1, iproc, nproc, old_grid, new_grid, calc_time, comm_time, step_reached_bottom);
            if (step_reached_bottom) {
                sim_reached_bottom = true;
            }
            steps++;
        }
        
        double sim_finish = MPI_Wtime();
        double sim_time = sim_finish-sim_start;
        MPI_Barrier(MPI_COMM_WORLD);
        // No need to reset old_grid and new_grid because they will be assigned new values in the next iteration anyway
        
        // Sum steps and bottom flag across runs
        total_steps_sum += steps;
        total_sim_time += sim_time;
        if (sim_reached_bottom) total_reached_bottom++;
    }
    
    // Average over runs (only rank 0 prints)
    if (iproc == 0) {
        double avg_steps = total_steps_sum / M;
        double avg_time = total_sim_time / M;
        double bottom_fraction = total_reached_bottom / M;
        
        std::cout << "Average number of steps: " << avg_steps << std::endl;
        std::cout << "Fire reached bottom in " << bottom_fraction * 100 << "% of runs" << std::endl;
        std::cout << "Average simulation time: " << avg_time << " seconds" << std::endl;
    }
    
    MPI_Finalize();
    return 0;

}
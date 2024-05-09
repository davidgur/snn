#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <future>

#define NUM_WALKERS 1000

struct State {
    std::vector<State*> neighbours;
    std::vector<double> transition_probabilities;

    int num_walkers;
    int num_incoming;
};
typedef std::vector<std::vector<State>> Mesh;
typedef std::vector<std::vector<double>> MeshValues;

double g(double x, double y, double dx) {
    if (std::abs(pow(x - 0.5, 2) + pow(y - 0.5, 2) - 0.005) < dx) {
        return 1;
    }

    if (std::abs(pow(x - 0.5, 2) + pow(y - 0.5, 2) - 0.1) < dx) {
        return -1;
    }

    return 0;
}


double u(double t, double x, double y, Mesh &states, double dx, double dt, double prev_t=0) {
    // Clear the number of walkers in each state
    if (prev_t == 0) {
        for (std::vector<State> &row : states) {
            for (State &state : row) {
                state.num_walkers = 0;
                state.num_incoming = 0;
            }
        }
        // Start the walkers at the initial position
        int init_x = (int)(x / dx);
        int init_y = (int)(y / dx);
        states[init_x][init_y].num_walkers = NUM_WALKERS;
    }


    // Simulate the walkers
    int num_timesteps = (int)((t - prev_t) / dt);

    for (int timestep = 0; timestep < num_timesteps; timestep++) {
        // Move the walkers
        for (int i = 0; i < states.size(); i++) {
            for (int j = 0; j < states[i].size(); j++) {
                for (int w = 0; w < states[i][j].num_walkers; w++) {
                    double r = (double) rand() / RAND_MAX;
                    double p = 0;
                    for (int k = 0; k < states[i][j].neighbours.size(); k++) {
                        p += states[i][j].transition_probabilities[k];
                        if (r < p) {
                            states[i][j].num_walkers--;
                            states[i][j].neighbours[k]->num_incoming++;
                            break;
                        }
                    }
                }
            }
        }

        // Update the number of walkers in each state
        for (int i = 0; i < states.size(); i++) {
            for (int j = 0; j < states[i].size(); j++) {
                states[i][j].num_walkers += states[i][j].num_incoming;
                states[i][j].num_incoming = 0;
            }
        }
    }

    double sum = 0;
    for (int i = 0; i < states.size(); i++) {
        for (int j = 0; j < states[i].size(); j++) {
            sum += states[i][j].num_walkers * g(i * dx, j * dx, dx);
        }
    }

    return sum / NUM_WALKERS;
}

int main(int argc, char *argv[]) {
    // The user passes the filename as a command-line argument
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <filename>" << std::endl;
        return 1;
    }

    // Load the model parameters from file
    std::string filename = argv[1];
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: could not open file " << filename << std::endl;
        return 1;
    }

    // Read the model parameters
    // The first line contains the length (in metres) of the domain
    double length;
    file >> length;

    // The second line contains the number of states
    int num_states;
    file >> num_states;

    // The third line contains dx
    double dx;
    file >> dx;

    // The fourth line contains dt
    double dt;
    file >> dt;

    // The remaining num_states * num_states lines contain the transition probabilities
    // Create the states
    Mesh states(num_states, std::vector<State>(num_states));
    for (int i = 0; i < num_states; i++) {
        for (int j = 0; j < num_states; j++) {
            // left
            double p_left;
            file >> p_left;
            if (i > 0) {
                states[i][j].transition_probabilities.push_back(p_left);
                states[i][j].neighbours.push_back(&states[i - 1][j]);
            } 

            // right
            double p_right;
            file >> p_right;
            if (i < num_states - 1) {
                states[i][j].transition_probabilities.push_back(p_right);
                states[i][j].neighbours.push_back(&states[i + 1][j]);
            }

            // up
            double p_up;
            file >> p_up;
            if (j < num_states - 1) {
                states[i][j].transition_probabilities.push_back(p_up);
                states[i][j].neighbours.push_back(&states[i][j + 1]);
            }

            // down
            double p_down;
            file >> p_down;
            if (j > 0) {
                states[i][j].transition_probabilities.push_back(p_down);
                states[i][j].neighbours.push_back(&states[i][j - 1]);
            }

            // stay
            double p_stay;
            file >> p_stay;
            states[i][j].transition_probabilities.push_back(p_stay);
            states[i][j].neighbours.push_back(&states[i][j]);
        }
    }

    // Close the file
    file.close();

    // Solve the PDE
    double t = 100;
    int num_steps = (int)(t / dt);

    std::vector<MeshValues> u_values(num_steps, MeshValues(num_states, std::vector<double>(num_states)));
    for (int i = 0; i < num_states; i++) {
        for (int j = 0; j < num_states; j++) {
            std::cerr << i << " / " << num_states << " (" << j << " / " << num_states << ")\r";
            for (int k = 0; k < num_steps; k++) {
                u_values[k][j][i] = u(k * dt, i * dx, j * dx, states, dx, dt, (k == 0) ? 0 : (k - 1) * dt);
            }
        }        
    }

    // Output the solution
    for (int k = 0; k < num_steps; k++) {
        for (int i = 0; i < num_states; i++) {
            for (int j = 0; j < num_states; j++) {
                std::cout << u_values[k][j][i] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    return 0;
}

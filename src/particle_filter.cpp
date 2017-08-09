/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"
#include "utils.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
    //   x, y, theta and their uncertainties from GPS) and all weights to 1. 
    // Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    num_particles = 100;

    // initialize particles and weights vectors
    particles.clear();
    particles.reserve(num_particles);

    weights.clear();
    weights.reserve(num_particles);

    // add noise component
    std::default_random_engine gen;

    std::normal_distribution<double> noise_x_dist(0, std[0]);
    std::normal_distribution<double> noise_y_dist(0, std[1]);
    std::normal_distribution<double> noise_theta_dist(0, std[2]);

    for (int i = 0; i < num_particles; ++i) {
        Particle p;
        p.id = i;
        p.x = x + noise_x_dist(gen);
        p.y = y + noise_y_dist(gen);
        p.theta = normalize(theta + noise_theta_dist(gen));
        p.weight = 1;

        particles.push_back(p);
        weights.push_back(p.weight);
    }

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    // TODO: Add measurements to each particle and add random Gaussian noise.
    // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/
    std::default_random_engine gen; // TODO(mike): Maybe extract gen to objec??

    std::normal_distribution<double> noise_x_dist(0, std_pos[0]);
    std::normal_distribution<double> noise_y_dist(0, std_pos[1]);
    std::normal_distribution<double> noise_theta_dist(0, std_pos[2]);

    double x_noise = noise_x_dist(gen);
    double y_noise = noise_y_dist(gen);
    double yaw_noise = noise_theta_dist(gen);

    if (fabs(yaw_rate) < 1e-6) {
        for (int i = 0; i < num_particles; ++i) {
            double delta_x = velocity * delta_t * cos(particles[i].theta) + x_noise;
            double delta_y = velocity * delta_t * sin(particles[i].theta) + y_noise;
            particles[i].x += delta_x;
            particles[i].y += delta_y;
            particles[i].theta = normalize(particles[i].theta + yaw_noise);

        }
    } else {
        for (int i = 0; i < num_particles; ++i) {
            particles[i].x += (velocity / yaw_rate) * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
            particles[i].x += x_noise;
            particles[i].y += (velocity / yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
            particles[i].y += y_noise;
            particles[i].theta = normalize(particles[i].theta + yaw_noise + yaw_rate * delta_t);
        }
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
    // TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
    //   observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
    //   implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
        std::vector<LandmarkObs> observations, Map map_landmarks) {
    // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
    //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
    //   according to the MAP'S coordinate system. You will need to transform between the two systems.
    //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
    //   The following is a good resource for the theory:
    //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
    //   and the following is a good resource for the actual equation to implement (look at equation 
    //   3.33
    //   http://planning.cs.uiuc.edu/node99.html

    double sum = 0.0;

    for (int i = 0; i < num_particles; ++i) {
        Particle p = particles[i];

        // remap observations from vehicle coordinate system to map coordinate system
        for (int obs_idx = 0; obs_idx < observations.size(); ++obs_idx) {
            LandmarkObs obs = observations[i];
            double x_map = cos(p.theta) * obs.x - sin(p.theta) * obs.y + p.x;
            double y_map = sin(p.theta) * obs.x + cos(p.theta) * obs.y + p.y;
            observations[i].x = x_map;
            observations[i].y = y_map;
        }

        // choose relevant observations
        std::vector<LandmarkObs> relevant_landmarks;
        for (auto landmark : map_landmarks.landmark_list) {
            double dist_ = dist(p.x, p.y, landmark.x_f, landmark.y_f);
            if (dist_ < sensor_range) {
                LandmarkObs obs = { landmark.x_f, landmark.y_f, landmark.id_i };
                relevant_landmarks.push_back(obs);
            }
        }

        for (auto obs : observations) { 

            // Identify the closest landmark
            double min_dist = sensor_range;
            LandmarkObs closest_landmark;
            bool closest_landmark_exists = false;
            for (auto landmark : relevant_landmarks) {
                double dist_ = dist(obs.x, obs.y, landmark.x, landmark.y);

                if (dist_ < min_dist) {
                    closest_landmark = landmark;
                    min_dist = dist_;
                    closest_landmark_exists = true;
                }
            }


            // Update the particle weight

            double x_diff, y_diff;

            if (closest_landmark_exists) {
                x_diff = closest_landmark.x - obs.x;
                y_diff = closest_landmark.y - obs.y;
            } else { 
                x_diff = y_diff = sensor_range;
            }

            double exp_arg = -(
                (x_diff * x_diff) / (2 * std_landmark[0] * std_landmark[0]) + 
                (y_diff * y_diff) / (2 * std_landmark[1] * std_landmark[1]) 
            );

            // TODO(Mike) denominator looks suspicious...
            double probability = exp(exp_arg) / sqrt(2 * M_PI * std_landmark[0] * std_landmark[1]);
            p.weight *= probability;
        }
    }


}

void ParticleFilter::resample() {
    // TODO: Resample particles with replacement with probability proportional to their weight. 
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    std::vector<Particle> new_particles;
    new_particles.reserve(num_particles);

    std::default_random_engine gen;
    std::discrete_distribution<int> particle_distribution(weights.begin(), weights.end());

    for (int i = 0; i < num_particles; ++i) {
        new_particles[i] = particles[particle_distribution(gen)];
    }

    particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    // Clear the previous associations
    particle.associations.clear();
    particle.sense_x.clear();
    particle.sense_y.clear();

    particle.associations = associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
    vector<int> v = best.associations;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
    vector<double> v = best.sense_x;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
    vector<double> v = best.sense_y;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

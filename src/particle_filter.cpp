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

    num_particles = 1000;

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
        p.theta = theta + noise_theta_dist(gen);
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


    for (int i = 0; i < num_particles; ++i) {
        double delta_x, delta_y;
        if (fabs(yaw_rate) < 1e-20) {
            delta_x = velocity * delta_t * cos(particles[i].theta);
            delta_y = velocity * delta_t * sin(particles[i].theta);
        } else {
            delta_x = (velocity / yaw_rate) * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
            delta_y = (velocity / yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
        }

        double x_noise = noise_x_dist(gen);
        double y_noise = noise_y_dist(gen);
        double yaw_noise = noise_theta_dist(gen);

        particles[i].x += delta_x + x_noise;
        particles[i].y += delta_y + y_noise;
        particles[i].theta = particles[i].theta + yaw_rate * delta_t + yaw_noise;
    }
}

void ParticleFilter::dataAssociation(std::vector<Map::single_landmark_s> predicted, std::vector<LandmarkObs>& observations) {
    // TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
    //   observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
    //   implement this method and use it as a helper during the updateWeights phase.

    for (int obs_idx = 0; obs_idx < observations.size(); ++obs_idx) {
        LandmarkObs *obs = &observations[obs_idx]; // Can't be written &?
        double min_so_far = 1e10;
        for (int i = 0; i < predicted.size(); ++i) {
            double distance = dist(
                    predicted[i].x_f, predicted[i].y_f,
                    obs->x, obs->y
            );
            if (distance < min_so_far) {
                obs->id = i;
                min_so_far = distance;
            }
        }
    }
}

void ParticleFilter::remap(std::vector<LandmarkObs> &observations, const Particle &p) {
    // remap observations from vehicle coordinate system to map coordinate system
    for (int obs_idx = 0; obs_idx < observations.size(); ++obs_idx) {
        LandmarkObs obs = observations[obs_idx];

        double x_map = cos(p.theta) * obs.x - sin(p.theta) * obs.y + p.x;
        double y_map = sin(p.theta) * obs.x + cos(p.theta) * obs.y + p.y;

        observations[obs_idx].x = x_map;
        observations[obs_idx].y = y_map;
    }
}


static double gaussianPdf(double ux, double uy, double x, double y, double std_x, double std_y) {

    double denom = 2 * M_PI * std_x * std_y;
    double exp_arg_x = -pow(x - ux, 2) / 2. / pow(std_x, 2);
    double exp_arg_y = -pow(y - uy, 2) / 2. / pow(std_y, 2);

    return exp(exp_arg_x + exp_arg_y) / denom;
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

    vector<Map::single_landmark_s> landmarks = map_landmarks.landmark_list;

    for (int i = 0; i < num_particles; ++i) {
        vector<LandmarkObs> remapped_obs(observations);
        remap(remapped_obs, particles[i]);
        dataAssociation(landmarks, remapped_obs);

        double weight = 1.0;

        for (auto obs : remapped_obs) {
            double landmark_x = landmarks[obs.id].x_f;
            double landmark_y = landmarks[obs.id].y_f;

            double prob = gaussianPdf(landmark_x, landmark_y, obs.x, obs.y, std_landmark[0], std_landmark[1]);
            weight *= prob;
        }

        particles[i].weight = weight;
        weights[i] = weight;
    }
}

void ParticleFilter::resample() {
    // TODO: Resample particles with replacement with probability proportional to their weight. 
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    std::vector<Particle> new_particles(num_particles);

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

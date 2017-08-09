#include <math.h>

double normalize(double angle) {
    return fmod(angle, 2.0 * M_PI);
}

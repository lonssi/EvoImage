
## EvoImage

### What is EvoImage?

EvoImage is an C++ application that uses a genetic algorithm to reconstruct a reference image using drawing primitives.

The following drawing primitives can be used:
- triangle
- circle
- ellipse
- rectangle
- line

The drawing of the images is implemented using OpenCV.
The project uses OpenMP to run the creation of new individuals, their evaluation and selection in parallel.

### Requirements

The following are required to run the project:

- `docker`
- `docker-compose`

### Getting started

To start the development environment run the `run-dev-environment.sh` script.

Example:
`./evoimage image.jpg rectangle`

### License

This project is distributed under the [MIT License](http://opensource.org/licenses/MIT).

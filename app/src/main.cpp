#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <vector>
#include <random>
#include <functional>
#include <limits>
#include <memory>

#include <omp.h>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "types.hpp"

using namespace cv;
using namespace std;


double standardDeviation(const vector<double>& samples, double avg) {
	double accum = 0.0;
	for_each(begin(samples), end(samples), [&](const double d) {
		accum += (d - avg) * (d - avg);
	});
	return sqrt(accum / (samples.size() - 1));
}

void writeCSV(size_t gen, double& mean, double& dev, double& best, string filename) {
	ofstream myfile;
	myfile.open(filename, ios_base::app);
	myfile << gen << ", " << mean << ", " << dev << ", " << best << endl;
	myfile.close();
}

class Evolver {

	public:

		Evolver(
			const Mat& _reference,
			float _fx,
			float _fxq,
			size_t _pop_size,
			float _selection,
			size_t _gen_max,
			size_t _n_shapes,
			float _m,
			float _mdev,
			string _fp,
			string _shape
		) :
			reference(_reference),
			fx(_fx),
			fxq(_fxq),
			pop_size(_pop_size),
			selection(_selection),
			gen_max(_gen_max),
			n_shapes(_n_shapes),
			m(_m),
			mdev(_mdev),
			fp(_fp),
			shape(_shape)
		{

			// Create working size image
			resize(reference, reference_resized, Size(), fx, fx, INTER_AREA);
			size_work = reference_resized.size();

			// Initialize rand engines
			int n_threads = max(1, omp_get_max_threads());
			for (int i = 0; i < n_threads; ++i) {
				random_device SeedDevice;
				rand_engines.push_back(mt19937(SeedDevice()));
			}

			remove(fp.c_str());
		};

		void evolve() {

			imshow("Original image", reference);
			imshow("Working resolution", reference_resized);
			waitKey(1);

			// Initialize population
			cout << "Creating initial population" << endl;
			vector<Individual> population(pop_size);
			for (size_t i = 0; i < pop_size; i++) {
				population[i] = createIndividual(size_work, n_shapes);
			}

			// Evaluate population
			cout << "Evaluating initial population" << endl;
			for (size_t i = 0; i < pop_size; i++) {
				population[i].drawImage();
				population[i].evaluate(reference_resized);
			}

			// main evolution loop
			cout << "Starting main loop" << endl;
			size_t g = 0;
			while (true) {

				// create child population
				vector<Individual> population_child(pop_size);
#pragma omp parallel for
				for (size_t i = 0; i < pop_size; i++) {
					const Individual& i_a = population[tournament_selection(population, pop_size * selection)];
					const Individual& i_b = population[tournament_selection(population, pop_size * selection)];
					Individual child = createChild(i_a, i_b, n_shapes, m, mdev);
					child.drawImage();
					child.evaluate(reference_resized);
					population_child[i] = child;
				}

				// selection
				vector<Individual> population_surviving(pop_size);
#pragma omp parallel for
				for (size_t i = 0; i < pop_size; i++) {
					if (population[i].getScore() > population_child[i].getScore()) {
						population_surviving[i] = population[i];
					} else {
						population_surviving[i] = population_child[i];
					}
				}
				population.clear();
				population = population_surviving;

				// Find most fit individual
				vector<double> scores;
				double best_score = -1;
				for (size_t i = 0; i < pop_size; i++) {
					Individual ind = population[i];
					double score = ind.getScore();
					scores.push_back(score);
					if (score > best_score) {
						best_score = score;
						best = ind;
					}
				}

				double mean = accumulate(begin(scores), end(scores), 0.0) / scores.size();
				double dev = standardDeviation(scores, mean);

				cout << "#################################" << endl;
				cout << "generation: " << g + 1 << "/" << gen_max << endl;
				cout << "population size: " << pop_size << endl;
				cout << "standard deviation: " << dev << endl;
				cout << "mean fitness: " << mean << endl;
				cout << "best fitness: " << best.getScore() << endl;

				best.drawImageQuality((1 / fx) * fxq);
				imshow("Current best", best.getHighResImage());
				waitKey(1);

				writeCSV(g, mean, dev, best_score, "/app/data.csv");

				if (g >= gen_max - 1 || dev < 1e-10)
					break;

				g++;
			}
		}

		Mat getBestImage(float scale) {
			best.drawImageQuality((1 / fx) * fxq);
			return best.getHighResImage();
		}

	private:

		Mat reference;
		float fx;
		float fxq;
		size_t pop_size;
		float selection;
		size_t gen_max;
		size_t n_shapes;
		float m = 0.01;
		float mdev = 0.05;
		string fp;
		string shape;

		Mat reference_resized;
		Size size_work;

		Individual best;

		vector<mt19937> rand_engines;

		int randomInt(int min, int max) {
			uniform_int_distribution<int> uid(min, max);
			return uid(rand_engines[omp_get_thread_num()]);
		}

		int randomIntNormal(float mean, float dev) {
			normal_distribution<float> nd(mean, dev);
			return (int) round(nd(rand_engines[omp_get_thread_num()]));
		}

		float randomFloat() {
			uniform_real_distribution<float> urd(0, 1);
			return urd(rand_engines[omp_get_thread_num()]);
		}

		inline Attribute createAttribute(int min, int max) {
			return Attribute(randomInt(min, max), min, max);
		}

		shared_ptr<Shape> createShape(const Size& size) {

			int wlim = size.width;
			int hlim = size.height;

			vector<Attribute> attributes = {
				createAttribute(0, 255),
				createAttribute(0, 255),
				createAttribute(0, 255),
				createAttribute(5, 75)
			};

			if (shape == "triangle") {

				attributes.push_back(createAttribute(0, wlim));
				attributes.push_back(createAttribute(0, hlim));
				attributes.push_back(createAttribute(0, wlim));
				attributes.push_back(createAttribute(0, hlim));
				attributes.push_back(createAttribute(0, wlim));
				attributes.push_back(createAttribute(0, hlim));

				return make_shared<Triangle>(attributes);

			} else if (shape == "circle") {

				int rmin = cvRound(0.1 * wlim);
				int rmax = cvRound(0.5 * wlim);
				attributes.push_back(createAttribute(0, wlim));
				attributes.push_back(createAttribute(0, hlim));
				attributes.push_back(createAttribute(rmin, rmax));

				return make_shared<Circle>(attributes);

			} else if (shape == "ellipse") {

				attributes.push_back(createAttribute(0, wlim));
				attributes.push_back(createAttribute(0, hlim));
				attributes.push_back(createAttribute(1, 0.5 * wlim));
				attributes.push_back(createAttribute(1, 0.5 * wlim));
				attributes.push_back(createAttribute(-180, 180));

				return make_shared<Ellipse>(attributes);

			} else if (shape == "rectangle") {

				attributes.push_back(createAttribute(0, wlim));
				attributes.push_back(createAttribute(0, hlim));
				attributes.push_back(createAttribute(0, wlim));
				attributes.push_back(createAttribute(0, hlim));

				return make_shared<Rectangle>(attributes);

			} else if (shape == "line") {

				int wmin = cvRound(0.01 * wlim);
				int wmax = cvRound(0.2 * wlim);

				attributes.push_back(createAttribute(0, wlim));
				attributes.push_back(createAttribute(0, hlim));
				attributes.push_back(createAttribute(0, wlim));
				attributes.push_back(createAttribute(0, hlim));
				attributes.push_back(createAttribute(wmin, wmax));

				return make_shared<Line>(attributes);

			} else {
				throw logic_error("Unsupported shape type");
			}
		}

		Individual createIndividual(const Size& size, size_t n_shapes) {
			vector<shared_ptr<Shape>> shapes(n_shapes);
			for (size_t i = 0; i < n_shapes; i++)
				shapes[i] = createShape(size);
			return Individual(size, shapes);
		}

		size_t tournament_selection(const vector<Individual>& population, size_t k) {

			size_t n = population.size();
			size_t best_index = randomInt(0, n - 1);
			Individual best = population[best_index];

			for (size_t i = 0; i < k; i++) {
				size_t index = randomInt(0, n - 1);
				if (population[index].getScore() > best.getScore()) {
					best = population[index];
					best_index = index;
				}
			}

			return best_index;
		}

		Individual createChild(const Individual& p1, const Individual& p2, size_t n_shapes, float M, float mdev) {

			Size sz = p1.getSize();
			vector<shared_ptr<Shape>> shapes(n_shapes);

			for (size_t i = 0; i < n_shapes; i++) {

				vector<Attribute> as1 = p1.getShape(i)->getAttributes();
				vector<Attribute> as2 = p2.getShape(i)->getAttributes();

				size_t n_attributes = as1.size();
				vector<Attribute> attributes(n_attributes);

				for (size_t j = 0; j < n_attributes; j++) {

					Attribute a1 = as1[j];
					Attribute a2 = as2[j];
					Attribute attribute = a1.clone();

					int value = (randomFloat() < 0.5) ? a1.v() : a2.v();

					if (randomFloat() < M) {
						int min = attribute.getMin();
						int max = attribute.getMax();
						int range = max - min;
						float dev = mdev * range;
						value += randomIntNormal(0, dev);
					}

					attribute.updateValue(value);
					attributes[j] = attribute;
				}

				if (shape == "triangle") {
					shapes[i] = make_shared<Triangle>(attributes);
				} else if (shape == "circle") {
					shapes[i] = make_shared<Circle>(attributes);
				} else if (shape == "ellipse") {
					shapes[i] = make_shared<Ellipse>(attributes);
				} else if (shape == "rectangle") {
					shapes[i] = make_shared<Rectangle>(attributes);
				} else if (shape == "line") {
					shapes[i] = make_shared<Line>(attributes);
				} else {
					throw logic_error("Unsupported shape type");
				}
			}

			return Individual(sz, shapes);
		}
};

bool shapeIsValid(const string& shape_user) {

	vector<string> shapes = {
		"triangle",
		"circle",
		"ellipse",
		"rectangle",
		"line"
	};

	for (const auto shape : shapes) {
		if (!shape.compare(shape_user)) {
			return true;
		}
	}

	return false;
}

int main(int argc, char** argv) {

	if (argc < 3) {
		cout << "Usage:" << endl;
		cout << argv[0] << " <image_path> <triangle|circle|ellipse|rectangle|line>" << endl;
		return 1;
	}

	Mat reference = imread(argv[1]);

	// Check that the image exists
	if (!reference.data) {
		cout << "Could not open image." << endl;
		return 1;
	}

	string shape_user = argv[2];

	// Check that the shape type is valid
	if (!shapeIsValid(shape_user)) {
		cout << "Invalid shape type." << endl;
		return 1;
	}

	size_t reference_max_width = 400;
	float fx_init = reference_max_width / (float) reference.cols;
	resize(reference, reference, Size(), fx_init, fx_init, INTER_AREA);

	size_t target_width = 75;
	float fx = target_width / (float) reference.cols;
	float fxq = 1.0;

	string filepath = "/app/data.csv";

	Evolver evolver = Evolver(
		reference,		// reference image
		fx,				// scale of internal resolution
		fxq,			// scale of presentation quality
		50,				// population size
		0.15,			// selection amount in tournament selection
		10000,			// max number of generations
		100,			// number of shapes
		0.01,			// mutation chance
		0.05,			// mutation amount
		filepath,		// result log filepath
		shape_user		// shape type
	);

	evolver.evolve();

	Mat best = evolver.getBestImage(fxq);
	imshow("best", best);
	waitKey();

	return 0;
}

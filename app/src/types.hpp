#ifndef TYPES
#define TYPES

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <vector>
#include <random>
#include <functional>
#include <limits>
#include <memory>

using namespace std;
using namespace cv;


double fitnessFunction(const Mat& image, const Mat& reference) {
	double diff = 0;
	for (int col = 0; col < image.cols; col++) {
		for (int row = 0; row < image.rows; row++) {
			Point p = Point(col, row);
			Vec<int, 3> c1 = image.at<Vec3b>(p);
			Vec<int, 3> c2 = reference.at<Vec3b>(p);
			diff += pow(c1[0] - c2[0], 2);
			diff += pow(c1[1] - c2[1], 2);
			diff += pow(c1[2] - c2[2], 2);
		}
	}
	return 1 - (diff / (image.cols * image.rows * 3 * 255 * 255));
}

class Attribute {

	public:

		Attribute() {};
		Attribute(int _value, int _min, int _max) : value(_value), min(_min), max(_max) {};
		Attribute clone() { return Attribute(value, min, max); };
		int v() { return value; };
		void updateValue(int _value) { value = clip(_value, min, max); };
		int getMin() { return min; };
		int getMax() { return max; };

	private:

		int value;
		int min;
		int max;

		int clip(int val, int a, int b) {
			if (val < a)
				return a;
			if (val > b)
				return b;
			return val;
		}
};

class Shape {

	public:

		Shape(const vector<Attribute>& _x) : x(_x) {};
		virtual ~Shape() {};

		// Getters
		virtual int getAttribute(int i) { return x[i].v(); };
		virtual vector<Attribute> getAttributes() { return x; };
		virtual size_t getSize() { return x.size(); };

		// Setters
		virtual void setAttribute(int i, int val) { x[i].updateValue(val); };

		// Methods
		virtual void drawShape(Mat& img, float scale, bool AA) = 0;

	protected:
		vector<Attribute> x;
};

class Triangle : public Shape {

	public:

		explicit Triangle(const vector<Attribute>& _x) : Shape(_x) {};

		virtual void drawShape(Mat& img, float scale, bool AA) {
			Mat canvas = img.clone();
			int mode = (AA) ? CV_AA : 8;
			drawContours(canvas, vector<vector<Point> >(1, getVector(scale)), -1, getColor(), -1, mode);
			float transparency = (float) getTransparency();
			float alpha = transparency / 100;
			float beta = 1 - alpha;
			addWeighted(canvas, alpha, img, beta, 0, img);
		};

	private:

		Scalar getColor() { return Scalar(x[0].v(), x[1].v(), x[2].v()); };
		int getTransparency() { return x[3].v(); };

		Point p1() { return Point(x[4].v(), x[5].v()); };
		Point p2() { return Point(x[6].v(), x[7].v()); };
		Point p3() { return Point(x[8].v(), x[9].v()); };
		vector<Point> getVector(float scale) { return { p1() * scale, p2() * scale, p3() * scale }; };
};

class Circle : public Shape {

	public:

		explicit Circle(const vector<Attribute>& _x) : Shape(_x) {};

		virtual void drawShape(Mat& img, float scale, bool AA) {
			Mat canvas = img.clone();
			int mode = (AA) ? CV_AA : 8;
			circle(canvas, getCenter() * scale, getRadius() * scale, getColor(), -1, mode);
			float transparency = (float) getTransparency();
			float alpha = transparency / 100;
			float beta = 1 - alpha;
			addWeighted(canvas, alpha, img, beta, 0, img);
		};

	private:

		Scalar getColor() { return Scalar(x[0].v(), x[1].v(), x[2].v()); };
		int getTransparency() { return x[3].v(); };

		Point getCenter() { return Point(x[4].v(), x[5].v()); };
		int getRadius() { return x[6].v(); };
};

class Ellipse : public Shape {

	public:

		explicit Ellipse(const vector<Attribute>& _x) : Shape(_x) {};

		virtual void drawShape(Mat& img, float scale, bool AA) {
			Mat canvas = img.clone();
			int mode = (AA) ? CV_AA : 8;
			ellipse(canvas, getCenter() * scale, Size(getAxis1() * scale, getAxis2() * scale),
				getAngle(), 0, 360, getColor(), -1, mode);
			float transparency = (float) getTransparency();
			float alpha = transparency / 100;
			float beta = 1 - alpha;
			addWeighted(canvas, alpha, img, beta, 0, img);
		};

	private:

		Scalar getColor() { return Scalar(x[0].v(), x[1].v(), x[2].v()); };
		int getTransparency() { return x[3].v(); };

		Point getCenter() { return Point(x[4].v(), x[5].v()); };
		int getAxis1() { return x[6].v(); };
		int getAxis2() { return x[7].v(); };
		int getAngle() { return x[8].v(); };
};

class Rectangle : public Shape {

	public:

		explicit Rectangle(const vector<Attribute>& _x) : Shape(_x) {};

		virtual void drawShape(Mat& img, float scale, bool AA) {
			Mat canvas = img.clone();
			int mode = (AA) ? CV_AA : 8;
			rectangle(canvas, p1() * scale, p2() * scale, getColor(), -1, mode);
			float transparency = (float) getTransparency();
			float alpha = transparency / 100;
			float beta = 1 - alpha;
			addWeighted(canvas, alpha, img, beta, 0, img);
		};

	private:

		Scalar getColor() { return Scalar(x[0].v(), x[1].v(), x[2].v()); };
		int getTransparency() { return x[3].v(); };

		Point p1() { return Point(x[4].v(), x[5].v()); };
		Point p2() { return Point(x[6].v(), x[7].v()); };
};

class Line : public Shape {

	public:

		explicit Line(const vector<Attribute>& _x) : Shape(_x) {};

		virtual void drawShape(Mat& img, float scale, bool AA) {
			Mat canvas = img.clone();
			int mode = (AA) ? CV_AA : 8;
			line(canvas, p1() * scale, p2() * scale, getColor(), getWidth() * scale, mode);
			float transparency = (float) getTransparency();
			float alpha = transparency / 100;
			float beta = 1 - alpha;
			addWeighted(canvas, alpha, img, beta, 0, img);
		};

	private:

		Scalar getColor() { return Scalar(x[0].v(), x[1].v(), x[2].v()); };
		int getTransparency() { return x[3].v(); };

		Point p1() { return Point(x[4].v(), x[5].v()); };
		Point p2() { return Point(x[6].v(), x[7].v()); };

		int getWidth() { return x[8].v(); };
};

class Individual {

	public:

		Individual() {};
		Individual(const Size& _size, const vector<shared_ptr<Shape>>& _shapes) : size(_size), shapes(_shapes) {};

		void drawImage() {
			img = Mat(size, CV_8UC3, Scalar(0, 0, 0));
			for (const auto& shape : shapes) {
				shape->drawShape(img, 1, false);
			}
		}

		void drawImageQuality(float scale) {
			img_q = Mat(size.height * scale, size.width * scale, CV_8UC3, Scalar(0, 0, 0));
			for (const auto& shape : shapes) {
				shape->drawShape(img_q, scale, true);
			}
		}

		void evaluate(const Mat& reference) {
			score = fitnessFunction(img, reference);
		}

		// Getters
		Size getSize() const { return size; };
		int getAttr(int i, int j) const { return shapes[i]->getAttribute(j); };
		Mat getImage() const { return img; };
		Mat getHighResImage() const { return img_q; };
		shared_ptr<Shape> getShape(size_t i) const { return shapes[i]; };
		vector<shared_ptr<Shape>> getShapes() const { return shapes; };
		double getScore() const { return score; };

		// Setters
		void setAttr(int i, int j, int k) { shapes[i]->setAttribute(j, k); };
		void setScore(double _score) { score = _score; };
		void setShapes(vector<shared_ptr<Shape>> _shapes) { shapes = _shapes; };

	private:

		Mat img;
		Mat img_q;
		Size size;
		vector<shared_ptr<Shape>> shapes;
		double score;
};

#endif

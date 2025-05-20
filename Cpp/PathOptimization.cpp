#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>
#include <json.hpp>
#include <geos/geom/Coordinate.h>
#include <geos/geom/LineSegment.h>
#include <geos/geom/Point.h>
#include <geos/geom/GeometryFactory.h>
#include <geos/operation/distance/DistanceOp.h>

using json = nlohmann::json;
using namespace std;
using namespace geos::geom;
using namespace geos::operation::distance;

struct Point2D {
    double x, y;
};

/*--------------------------------------------------------------------------------------------------*/
// Load data from a JSON file and return a vector of points (2DStructs)
/*--------------------------------------------------------------------------------------------------*/
vector<Point2D> load_data(const string& filename) {
    ifstream file(filename);
    json data;
    file >> data;

	// Iterate through the json file and load the points into a 2D vector (list)
    vector<Point2D> result;
    for (const auto& item : data) {
        result.push_back({ item["x"], item["y"] });
    }
    return result;
}

/*--------------------------------------------------------------------------------------------------*/
/* Calculate distance between two given points(a, b) using GEOS library                             */
/*--------------------------------------------------------------------------------------------------*/
double compute_distance(const Coordinate& a, const Coordinate& b) {
    GeometryFactory::Ptr factory = GeometryFactory::create();
    unique_ptr<Point> p1(factory->createPoint(a));
    unique_ptr<Point> p2(factory->createPoint(b));
    return DistanceOp::distance(p1.get(), p2.get());
}

/*--------------------------------------------------------------------------------------------------*/
/* Compute the partial derivative gradient given a set of segments(A, B) and a set of parameters(t) */
/*--------------------------------------------------------------------------------------------------*/
vector<double> polyline_gradient(const Coordinate& S, const vector<Coordinate>& A, const vector<Coordinate>& B, const vector<double>& t) {
    size_t n = A.size();
    vector<Coordinate> P(n);
	vector<double> dL_dt(n, 0.0); //Initialize the gradient vector

	// Calculate the points P based on the segments A, B and the parameters t
    for (size_t i = 0; i < n; ++i) {
        P[i] = Coordinate(
            A[i].x + (B[i].x - A[i].x) * t[i],
            A[i].y + (B[i].y - A[i].y) * t[i]
        );
    }

	// Calculate the gradient of the total polyline length with respect to t
    for (size_t i = 0; i < n; ++i) {
        Coordinate Pi = P[i];
        Coordinate dPi(B[i].x - A[i].x, B[i].y - A[i].y);

		Coordinate prev = (i == 0) ? S : P[i - 1]; //Use S for the first point
		Coordinate diff_prev(Pi.x - prev.x, Pi.y - prev.y); //Calculate the difference vector
		double norm_prev = hypot(diff_prev.x, diff_prev.y); //Calculate the norm of the vector
		double dL_dt_prev = (norm_prev > 0) ? ((diff_prev.x * dPi.x + diff_prev.y * dPi.y) / norm_prev) : 0; //Calculate the partial derivative for the previous point

		// Calculate the partial derivative for the next point
        double dL_dt_next = 0;
        if (i < n - 1) {
            Coordinate next = P[i + 1];
            Coordinate diff_next(Pi.x - next.x, Pi.y - next.y);
            double norm_next = hypot(diff_next.x, diff_next.y);
            dL_dt_next = (norm_next > 0) ? ((diff_next.x * dPi.x + diff_next.y * dPi.y) / norm_next) : 0;
        }

		// Calculate the total partial derivative for the current point using the previous and next points
        dL_dt[i] = dL_dt_prev + dL_dt_next;
    }

    return dL_dt;
}

/*--------------------------------------------------------------------------------------------------*/
/* Save the points to a JSON file in the specified format (same as the input data)                  */
/*--------------------------------------------------------------------------------------------------*/
void save_as_json(const vector<Coordinate>& P, const string& filename) {
    json output = json::array();

    for (size_t i = 0; i < P.size(); ++i) {
        json point;
        point["label"] = "P" + to_string(i + 1);
        point["x"] = round(P[i].x * 1000.0) / 1000.0;
        point["y"] = round(P[i].y * 1000.0) / 1000.0;
        output.push_back(point);
    }

    ofstream file(filename);
    file << setw(2) << output << endl;
}

/*--------------------------------------------------------------------------------------------------*/
/* Main Program                                                                                     */
/*--------------------------------------------------------------------------------------------------*/
int main() {

	// Load the data from JSON files
    vector<Point2D> nodes_raw = load_data("OrcaShortestPath\\Data\\nodes.json");
    vector<Point2D> SG_raw = load_data("OrcaShortestPath\\Data\\from_to.json");

	// Convert the raw data into Coordinate objects
    vector<Coordinate> nodes, SG;
    for (const auto& pt : nodes_raw) {
		nodes.emplace_back(pt.x, pt.y); //Segments A and B (defined by the nodes)
    }
    for (const auto& pt : SG_raw) {
		SG.emplace_back(pt.x, pt.y); //Start and Goal points of the polyline
    }

	// Check if the number of nodes is even, if not, add the last node to the end of the list
    vector<Coordinate> A_nodes, B_nodes;
    for (size_t i = 0; i < nodes.size(); i += 2) {
        A_nodes.push_back(nodes[i]);
        if (i + 1 < nodes.size())
            B_nodes.push_back(nodes[i + 1]);
        else
            B_nodes.push_back(SG[1]);
    }
	// Append the goal point to connect the last segment of the polyline to the goal point
    A_nodes.push_back(SG[1]);
    B_nodes.push_back(SG[1]);

	// Initialize parameters for the optimization process
    size_t n_nodes = A_nodes.size();
	vector<double> t(n_nodes, 0.5); //Initialize the optimal point at the middle of the segment
	vector<Coordinate> P(n_nodes); //Initialize the points P based on the segments A, B and the parameters t

	//Define the gradient descent parameters
	double G = 0.02; // Gradient descent gain
	double tol = 1e-3; // Tolerance for convergence
    double dL = 100.0; // Large initial value
    double L_ans = 100.0;
	int iteration = 0; // Iteration counter

    while (dL > tol) {
		vector<double> dL_dt = polyline_gradient(SG[0], A_nodes, B_nodes, t); // Calculate the gradient of the total polyline length given the current intial points
        for (size_t i = 0; i < t.size(); ++i) {
			t[i] -= G * dL_dt[i]; // Update the parameters t using the gradient descent method
			t[i] = max(0.0, min(1.0, t[i])); // Ensure t is within the range [0, 1]
            P[i] = Coordinate(
                A_nodes[i].x + (B_nodes[i].x - A_nodes[i].x) * t[i],
                A_nodes[i].y + (B_nodes[i].y - A_nodes[i].y) * t[i]
            );
        }

		// Calculate the total length of the polyline
        double L = 0.0;
        Coordinate prev = SG[0];
        for (const auto& p : P) {
            L += compute_distance(prev, p);
            prev = p;
        }

		// Calculate the distance from the last point to the goal point
        dL = abs(L - L_ans);
        L_ans = L;
        iteration++;
    }

    cout << "Final length: " << fixed << setprecision(4) << L_ans << " after " << iteration << " iterations." << endl;

    save_as_json(P, "ShortestPath.json");

    return 0;
}

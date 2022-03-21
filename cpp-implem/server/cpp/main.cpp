#include "Poco/DigestStream.h"
#include "Poco/Net/HTTPRequest.h"
#include "Poco/Net/HTTPClientSession.h"
#include "Poco/Net/HTTPRequest.h"
#include "Poco/Net/HTTPResponse.h"
#include "Poco/StreamCopier.h"
#include "Poco/Path.h"
#include "Poco/URI.h"
#include "Poco/Exception.h"
#include "main.h"
#include "pistache/endpoint.h"
#include "pistache/http.h"
#include "pistache/router.h"
#include <sstream>

using namespace Pistache;
using namespace Rest;
using namespace Log;
using namespace Poco;
using namespace Net;
using namespace seal;
using namespace std;

vector<double> idfWeights;
vector<vector<double>> linearParams;
vector<double> interceptParams;

class CKKSHost {
	public: 
		CKKSHost(shared_ptr<SEALContext> ctx, PublicKey pk, RelinKeys rk, GaloisKeys gk) {
			poly_degree = 16384;
			scale = pow(2.0, 40);
			context = ctx;
			publicKey = pk;
			relinKeys = rk; 
			galoisKeys = gk;
		}

		Ciphertext doIdf(Ciphertext ct) {
			cout << "Calculating encrypted prediction" << endl;
			CKKSEncoder enc(context);
			Evaluator eval(context);

			cout << "Multiplying with idf coefficient" << endl;
			Plaintext idf;
			enc.encode(idfWeights, ct.scale(), idf);
			eval.multiply_plain_inplace(ct, idf);
			eval.relinearize_inplace(ct, relinKeys);
			eval.rescale_to_next_inplace(ct);
			return ct;
		}

		Ciphertext multiplyWeights(Ciphertext ct) {
			CKKSEncoder enc(context);
			Evaluator eval(context);
			cout << "Multiplying weights" << endl;
			vector<Ciphertext> output(linearParams.size());
			for (int i = 0 ; i < linearParams.size() ; i++) {
				Plaintext weights;
				enc.encode(linearParams[i], scale, weights);
				eval.mod_switch_to_inplace(weights, ct.parms_id());

				Ciphertext nthVector;
				eval.multiply_plain(ct, weights, nthVector);
				eval.relinearize_inplace(nthVector, relinKeys);
				eval.rescale_to_next_inplace(nthVector);

				sumValues(nthVector);

				enc.encode(interceptParams[i], nthVector.scale(), weights);
				eval.mod_switch_to_inplace(weights, nthVector.parms_id());
				eval.add_plain_inplace(nthVector, weights);
				output[i] = nthVector;
			}

			Ciphertext finalOutput;
			for (int i = 0; i < output.size(); i++) {
				Plaintext selector;
				Ciphertext select;
				if(i == 0) {
					enc.encode(vector<double>{1,0,0}, scale, selector);
					eval.mod_switch_to_inplace(selector, output[i].parms_id());
					eval.multiply_plain(output[i], selector, select);
					eval.rescale_to_next_inplace(output[i]);
					finalOutput = select;
				}

				if(i == 1) {
					enc.encode(vector<double>{0,1,0}, scale, selector);
					eval.mod_switch_to_inplace(selector, output[i].parms_id());
					eval.multiply_plain(output[i], selector, select);
					eval.add_inplace(finalOutput, select);
				}

				if(i == 2) {
					enc.encode(vector<double>{0,0,1}, scale, selector);
					eval.mod_switch_to_inplace(selector, output[i].parms_id());
					eval.multiply_plain(output[i], selector, select);
					eval.add_inplace(finalOutput, select);
				}
			}

			return finalOutput;
		}


	private:
		double scale;
		size_t poly_degree;
		shared_ptr<SEALContext> context;
		PublicKey publicKey;
		RelinKeys relinKeys;
		GaloisKeys galoisKeys;
		vector<double> idf;

		void sumValues(Ciphertext &vector) {
			Evaluator eval(context);
			Ciphertext rot; 
			for (unsigned i = poly_degree / 4; i >= 1 ; i = i / 2) {
				eval.rotate_vector(vector, i, galoisKeys, rot);
				eval.add_inplace(vector, rot);
			}
		}
};

class Server {
	public: 
		explicit Server(Address addr) : httpEndpoint(std::make_shared<Http::Endpoint>(addr)) {
			cout << "Created server instance with address: " << addr.host() << endl;
		}
		
		void initSeal() {
			EncryptionParameters parms(scheme_type::CKKS);

			size_t poly_modulus_degree = 16384;
			parms.set_poly_modulus_degree(poly_modulus_degree);
			parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, { 60, 40, 40, 40, 60}));
			double scale = pow(2.0, 40);

			context = SEALContext::Create(parms);
			print_parameters(context);
			cout << endl;
		}

		void init() {
			auto opts = Http::Endpoint::options()
				.threads(static_cast<int>(1))
				.maxRequestSize(1024 * 1024 * 1024);
			httpEndpoint->init(opts);
			setupRoutes();
		}

		void start() {
			httpEndpoint->setHandler(router.handler());
			httpEndpoint->serve();
		}

		shared_ptr<SEALContext> context;
		PublicKey publicKey;
		RelinKeys relinKeys;
		GaloisKeys galoisKeys;

	private: 
		void setupRoutes() {
			using namespace Rest;

			Routes::Post(router, "/keys/public", Routes::bind(&Server::setPK, this));
			Routes::Post(router, "/keys/relin", Routes::bind(&Server::setRK, this));
			Routes::Post(router, "/keys/galois", Routes::bind(&Server::setGK, this));
			Routes::Post(router, "/operations/square", Routes::bind(&Server::encryptedSquare, this));
			Routes::Post(router, "/operations/score", Routes::bind(&Server::calculateScores, this));
		}

		void setPK(const Rest::Request& request, Http::ResponseWriter response) {
			cout << "Setting public key with length: " << request.body().length() << endl;
			istringstream istr(request.body());
			publicKey.load(context, istr);
			cout << "Public key set!" << endl;
			response.send(Http::Code::Ok, "Public key set!");
		}
		
		void setRK(const Rest::Request& request, Http::ResponseWriter response) {
			cout << "Setting relinearization key with length: " << request.body().length() << endl;
			istringstream istr(request.body());
			relinKeys.load(context, istr);
			cout << "Relinearization key set!" << endl;
			response.send(Http::Code::Ok, "Relinearization key set!");
		}
		
		void setGK(const Rest::Request& request, Http::ResponseWriter response) {
			cout << "Setting galois key with length: " << request.body().length() << endl;
			istringstream istr(request.body());
			galoisKeys.load(context, istr);
			cout << "Galois key set!" << endl;
			response.send(Http::Code::Ok, "Galois key set!");
		}

		void encryptedSquare(const Rest::Request& request, Http::ResponseWriter response) {
			cout << "Squaring ciphertext with length: " << request.body().length() << endl;
			Ciphertext ct;
			istringstream istr(request.body());
			ct.load(context, istr);

			CKKSEncoder enc(context);
			Evaluator eval(context);
			Plaintext pt;
			enc.encode(2, pt);
			eval.multiply_plain_inplace(ct, pt);

			std::ostringstream stream;
			ct.save(stream);
			string streamBody = stream.str();

			cout << "Returning ciphertext with length: " << streamBody.length() << endl;
			response.send(Http::Code::Ok, streamBody);
		}

		void calculateScores(const Rest::Request& request, Http::ResponseWriter response) {
			cout << "Calculating scores for ciphertext with length: " << request.body().length() << endl;
			Ciphertext ct;
			istringstream istr(request.body());
			ct.load(context, istr);

			CKKSHost host(context, publicKey, relinKeys, galoisKeys);
			Ciphertext withIdf = host.doIdf(ct);
			Ciphertext withWeights = host.multiplyWeights(withIdf);

			std::ostringstream stream;
			withWeights.save(stream);
			string streamBody = stream.str();

			cout << "Returning comma-separated vector with length: " << streamBody.length() << endl;
			response.send(Http::Code::Ok, streamBody);
		}

    	std::shared_ptr<Http::Endpoint> httpEndpoint;
    	Rest::Router router;
};

int main(int argc, char** argv)
{
	requestParamsFromServer();

    Pistache::Address addr(Pistache::Ipv4::any(), Pistache::Port(9080));
    Server endpoints(addr);

	endpoints.initSeal();
	endpoints.init();
	endpoints.start();

	
}

void requestParamsFromServer() {
	try {
        // prepare session
        URI uri("http://127.0.0.1:5000/model");
        HTTPClientSession session(uri.getHost(), uri.getPort());

        // prepare path
        string path(uri.getPathAndQuery());
        if (path.empty()) path = "/";

        // send request
        HTTPRequest req(HTTPRequest::HTTP_GET, path, HTTPMessage::HTTP_1_1);
        session.sendRequest(req);

        // get response
        HTTPResponse res;
        cout << "Retrieving model parameters from server..." << endl;
        cout << res.getStatus() << " " << res.getReason() << endl;

        // print response
        istream &is = session.receiveResponse(res);

		string content {
			istreambuf_iterator<char>(is),
		    istreambuf_iterator<char>()
		};

		content = content.substr(18, content.size()-20);

		string paramString = content.substr(0, content.find("', 'intercept': '"));
		paramString = paramString.substr(1, paramString.size()-2);

		string intercept = content.substr(content.find("', 'intercept': '") + 17, content.size());
		intercept = intercept.substr(1, intercept.size()-2);

		vector<string> lines = split(paramString, ';');
		vector<vector<double>> output(lines.size());
		for (unsigned i=0; i < lines.size(); i++) {
			string unwrapped = lines[i];
			unwrapped.erase(std::remove(unwrapped.begin(), unwrapped.end(), '['), unwrapped.end());
			unwrapped.erase(std::remove(unwrapped.begin(), unwrapped.end(), ']'), unwrapped.end());
			vector<string> splitVector = split(unwrapped, ',');
			vector<double> castVector(splitVector.size());
			transform(splitVector.begin(), splitVector.end(), castVector.begin(), [](const std::string& val) {
				return std::stod(val);
			});
			output[i] = castVector;
		}
		linearParams = output;
		cout << "Initialized SVC weights with " << linearParams.size() << " vectors, each vector containing " << linearParams[0].size() << " parameters"<< endl;

		vector<string> splitVector = split(intercept, ',');
		vector<double> castVector(splitVector.size());
		transform(splitVector.begin(), splitVector.end(), castVector.begin(), [](const std::string& val) {
			return std::stod(val);
		});
		interceptParams = castVector;
		cout << "Initialized intercept with array of length " << interceptParams.size() << endl;
    } catch (Exception &ex) {
        cerr << ex.displayText() << endl;
        return;
    }

    try {
        // prepare session
        URI uri("http://127.0.0.1:5000/idf");
        HTTPClientSession session(uri.getHost(), uri.getPort());

        // prepare path
        string path(uri.getPathAndQuery());
        if (path.empty()) path = "/";

        // send request
        HTTPRequest req(HTTPRequest::HTTP_GET, path, HTTPMessage::HTTP_1_1);
        session.sendRequest(req);

        // get response
        HTTPResponse res;
        cout << "Retrieving IDF weights from server..." << endl;
        cout << res.getStatus() << " " << res.getReason() << endl;

        // print response
        istream &is = session.receiveResponse(res);

		string content {
			istreambuf_iterator<char>(is),
		    istreambuf_iterator<char>()
		}; 

		vector<string> vec = split(content, ' ');
		
		vector<double> output(vec.size());
		transform(vec.begin(), vec.end(), output.begin(), [](const std::string& val) {
			return std::stod(val);
		});
		
		idfWeights = output;
		cout << "Initialized IDF weights with array of length " << idfWeights.size() << endl;
    } catch (Exception &ex) {
        cerr << ex.displayText() << endl;
        return;
    }
}
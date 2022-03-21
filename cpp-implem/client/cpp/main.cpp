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

class Client {
	public: 
		explicit Client(Address addr) : httpEndpoint(std::make_shared<Http::Endpoint>(addr)) {
			cout << "Created client instance with address: " << addr.host() << endl;
		}
		
		void initSeal() {
			EncryptionParameters parms(scheme_type::CKKS);

			size_t poly_modulus_degree = 16384;
			parms.set_poly_modulus_degree(poly_modulus_degree);
			parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, { 60, 40, 40, 40, 60}));
			scale = pow(2.0, 40);

			context = SEALContext::Create(parms);
			print_parameters(context);
			cout << endl;
			
			KeyGenerator keygen(context);
			publicKey = keygen.public_key();
			secretKey = keygen.secret_key();
			relinKeys = keygen.relin_keys_local();
			galoisKeys = keygen.galois_keys_local();
			cout << "Galois key size: " << float(galoisKeys.save_size()) / (1024 * 1024) << endl;
		}

		void init() {
			auto opts = Http::Endpoint::options()
				.threads(static_cast<int>(1))
				.maxRequestSize(1024 * 1024 * 1024);
			httpEndpoint->init(opts);
			setupRoutes();
		}

		void initRemote() {
			// prepare session
			URI uri("http://127.0.0.1:9080");
			HTTPClientSession session(uri.getHost(), uri.getPort());

			{
				// prepare path
				string path(uri.getPathAndQuery());
				if (path.empty()) path = "/keys/public";
				std::ostringstream stream;
				publicKey.save(stream);
				string streamBody = stream.str();

				// send request
				HTTPRequest req(HTTPRequest::HTTP_POST, path, HTTPMessage::HTTP_1_1);
				req.setContentType("text/plain");
				req.setKeepAlive(true); // notice setKeepAlive is also called on session (above)
				req.setContentLength( streamBody.length() );

				cout << streamBody.length() << endl;
				
				std::ostream& myOStream = session.sendRequest(req); // sends request, returns open stream
				myOStream << streamBody;  // sends the body
				
				HTTPResponse res;
				// print response
				istream &is = session.receiveResponse(res);
				cout << res.getStatus() << " " << res.getReason() << endl;
				string content {
					istreambuf_iterator<char>(is),
					istreambuf_iterator<char>()
				};
			}

			{
				// prepare path
				string path(uri.getPathAndQuery());
				if (path.empty()) path = "/keys/relin";
				std::ostringstream stream;
				relinKeys.save(stream);
				string streamBody = stream.str();

				// send request
				HTTPRequest req(HTTPRequest::HTTP_POST, path, HTTPMessage::HTTP_1_1);
				req.setContentType("text/plain");
				req.setKeepAlive(true); // notice setKeepAlive is also called on session (above)

				req.setContentLength( streamBody.length() );

				cout << streamBody.length() << endl;
				
				std::ostream& myOStream = session.sendRequest(req); // sends request, returns open stream
				myOStream << streamBody;  // sends the body
				
				HTTPResponse res;
				// print response
				istream &is = session.receiveResponse(res);
				cout << res.getStatus() << " " << res.getReason() << endl;
				string content {
					istreambuf_iterator<char>(is),
					istreambuf_iterator<char>()
				};
				cout << content << endl;
			}

			{
				// prepare path
				string path(uri.getPathAndQuery());
				if (path.empty()) path = "/keys/galois";
				std::ostringstream stream;
				galoisKeys.save(stream);
				string streamBody = stream.str();

				// send request
				HTTPRequest req(HTTPRequest::HTTP_POST, path, HTTPMessage::HTTP_1_1);
				req.setContentType("text/plain");
				req.setKeepAlive(true); // notice setKeepAlive is also called on session (above)
				req.setContentLength( streamBody.length() );

				cout << streamBody.length() << endl;
				
				std::ostream& myOStream = session.sendRequest(req); // sends request, returns open stream
				myOStream << streamBody;  // sends the body
				
				HTTPResponse res;
				// print response
				istream &is = session.receiveResponse(res);
				cout << res.getStatus() << " " << res.getReason() << endl;
				string content {
					istreambuf_iterator<char>(is),
					istreambuf_iterator<char>()
				};
			}
		}

		void start() {
			httpEndpoint->setHandler(router.handler());
			httpEndpoint->serve();
		}

	private: 
		void setupRoutes() {
			using namespace Rest;

			Routes::Post(router, "/score", Routes::bind(&Client::requestScore, this));
		}

		void requestScore(const Rest::Request& request, Http::ResponseWriter response) {
			// prepare session
			vector<string> vec = split(request.body(), ',');
			cout << "Received request to score entry with body: " << vec.size() << endl;
		
			vector<double> output(vec.size());
			transform(vec.begin(), vec.end(), output.begin(), [](const std::string& val) {
				return std::stod(val);
			});

			URI uri("http://127.0.0.1:9080");
			HTTPClientSession session(uri.getHost(), uri.getPort());

			Encryptor encryptor(context, publicKey);
			Evaluator evaluator(context);
			Decryptor decryptor(context, secretKey);
			CKKSEncoder encoder(context);
			size_t slot_count = encoder.slot_count();

			Plaintext plain;
			encoder.encode(output, scale, plain);

			Ciphertext ciphertext;
			encryptor.encrypt(plain, ciphertext);

			// prepare path
			string path(uri.getPathAndQuery());
			if (path.empty()) path = "/operations/score";
			std::ostringstream stream;
			ciphertext.save(stream);
			string streamBody = stream.str();

			// send request
			HTTPRequest req(HTTPRequest::HTTP_POST, path, HTTPMessage::HTTP_1_1);
			req.setContentType("text/plain");
			req.setKeepAlive(true); // notice setKeepAlive is also called on session (above)

			req.setContentLength( streamBody.length() );

			cout << streamBody.length() << endl;
			
			std::ostream& myOStream = session.sendRequest(req); // sends request, returns open stream
			myOStream << streamBody;  // sends the body
			
			HTTPResponse res;
			// print response
			istream &is = session.receiveResponse(res);
			cout << res.getStatus() << " " << res.getReason() << endl;
			string content {
				istreambuf_iterator<char>(is),
				istreambuf_iterator<char>()
			};

			istringstream istr(content);
			ciphertext.load(context, istr);

			Plaintext plain_result;
			vector<double> result;

			decryptor.decrypt(ciphertext, plain_result);
			encoder.decode(plain_result, result);

			std::ostringstream vts; 
			for (int i = 0 ; i < 3 ; i++) {
				cout << result[i] << endl;
				vts << result[i];
				if(i != 2) {
					vts << ",";
				}
			}

			cout << "Returning comma-separated vector: " << vts.str() << endl;
			response.send(Http::Code::Ok, vts.str());
		}

    	std::shared_ptr<Http::Endpoint> httpEndpoint;
    	Rest::Router router;

		double scale;
		shared_ptr<SEALContext> context;
		PublicKey publicKey;
		SecretKey secretKey;
		RelinKeys relinKeys;
		GaloisKeys galoisKeys;
};

int main(int argc, char** argv)
{
    Pistache::Address addr(Pistache::Ipv4::any(), Pistache::Port(9081));
    Client client(addr);

	client.initSeal();
	client.init();
	client.initRemote();
	client.start();
	return 0;
}
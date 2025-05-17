import argparse
import joblib
from src.model import train
from src.predict import predict


def main():
    parser = argparse.ArgumentParser(description="Email phishing detection baseline")
    parser.add_argument('--train', metavar='CSV', help='Train model using dataset CSV')
    parser.add_argument('--predict', metavar='EML', help='Predict phishing for an email file')
    parser.add_argument('--model', default='models/phishing_model.joblib', help='Path to save/load the model')
    args = parser.parse_args()

    if args.train:
        model, acc = train(args.train)
        joblib.dump(model, args.model)
        print(f"Model saved to {args.model}")
        print(f"Accuracy: {acc:.4f}")
    elif args.predict:
        result = predict(args.model, args.predict)
        print(f"Is phishing: {result['is_phishing']}, confidence: {result['confidence']:.4f}")
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

import sys
import os
from flask import Flask, request, jsonify

# Add parent directory to sys.path to allow importing calculator
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calculator.profit import estimate_profit

app = Flask(__name__)

@app.route('/profit', methods=['POST'])
def calculate_profit_endpoint():
    """
    Step 13: POST /profit API endpoint
    Expects JSON payload:
    {
        "crop": "Groundnut",
        "yield": 3.8,
        "farmArea": 2
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid request. JSON payload expected."}), 400

        # Validate required fields
        required_fields = ["crop", "yield", "farmArea"]
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({"error": f"Missing required field(s): {', '.join(missing_fields)}"}), 400

        crop = data.get("crop")
        yield_val = data.get("yield")
        farm_area = data.get("farmArea")
        district = data.get("district")
        recommended_crop = data.get("recommendedCrop")

        # Basic type validation
        try:
            yield_val = float(yield_val)
            farm_area = float(farm_area)
        except (ValueError, TypeError):
            return jsonify({"error": "'yield' and 'farmArea' must be valid numeric values."}), 400

        if farm_area <= 0 or yield_val <= 0:
            return jsonify({"error": "'yield' and 'farmArea' must be greater than zero."}), 400

        # Calculate profit report
        report = estimate_profit(
            crop=crop,
            yield_val=yield_val,
            farm_area=farm_area,
            district=district,
            recommended_crop=recommended_crop
        )

        return jsonify(report), 200

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 404
    except Exception as e:
        return jsonify({"error": "An error occurred during profit calculation.", "details": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "service": "profit-estimation-api"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

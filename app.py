from flask import Flask, render_template, url_for

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

# @app.route('/customer')
# def customer():
#     return 'Hello'

@app.route('/customer')
def customer():
    # Run the app.py file and capture its output
    import subprocess
    result = subprocess.run(['python', 'customer.py'], capture_output=True, text=True)

    # Pass the output to the template and render it
    return render_template('Customer_redirect_page.html', output=result.stdout)

@app.route('/organization')
def organization():
    # Run the app.py file and capture its output    
    import subprocess
    result = subprocess.run(['python', 'organization.py'], capture_output=True, text=True)

    # Pass the output to the template and render it
    return render_template('Organization_redirect_page.html', output=result.stdout)

# @app.route('/read')
# def get_file():
#     with open("customer.py", "r") as f:
#         content = f.read()
#     return content

if __name__ == '__main__':
    app.run(debug=True)

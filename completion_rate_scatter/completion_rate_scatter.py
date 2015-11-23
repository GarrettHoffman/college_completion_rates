import flask


#---------- URLS AND WEB PAGES -------------#

# Initialize the app
app = flask.Flask(__name__, static_url_path='/static')

# Homepage
@app.route("/")
def viz_page():
    """
    serve our visualization page, college_completion_scatter.html
    """
    with open("college_completion_scatter.html", 'r') as viz_file:
        return viz_file.read()

#Sending our data
@app.route('/data/')
def data():
    return "<a href=%s>file</a>" % flask.url_for('static', filename='d3_data.json')

#--------- RUN WEB APP SERVER ------------#

# Start the app server on port 80
# (The default website port)

if __name__ == '__main__':
    app.run(debug=True,
        host = "0.0.0.0",
        port = 80
    )

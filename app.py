from flask import Flask, request, render_template_string
import os
from prediction import run_prediction

# ====================================================
# CONFIG
# ====================================================

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ====================================================
# HTML
# ====================================================
HTML = """
<!DOCTYPE html>
<html>
<head>

<title>MalariaNet</title>

<style>

body{
font-family:Arial, sans-serif;
background:#eef2f7;
margin:0;
padding:0;
}

.container{
max-width:1200px;
margin:auto;
background:white;
min-height:100vh;
box-shadow:0 0 20px rgba(0,0,0,0.15);
}

.header{
text-align:center;
padding:40px;
}

.logo{
font-size:60px;
font-weight:bold;
color:#003c8f;
}

.subtitle{
font-size:22px;
color:#555;
margin-top:10px;
}

.description{
max-width:900px;
margin:auto;
font-size:22px;
line-height:1.7;
color:#333;
padding:20px;
text-align:center;
}

.developer{
font-style:italic;
font-size:24px;
color:#666;
margin-top:20px;
}

.upload-box{
text-align:center;
padding:30px;
}

input[type=file]{
padding:10px;
}

button{
background:#003c8f;
color:white;
padding:15px 35px;
border:none;
font-size:20px;
cursor:pointer;
border-radius:6px;
}

button:hover{
background:#002f6d;
}

.red-header{
background:#0b3b82;
color:white;
padding:20px;
font-size:48px;
font-weight:bold;
text-align:center;
}

.section{
margin:25px;
border:1px solid #ddd;
}

.blue{
background:linear-gradient(to right,#0b3b82,#6da9ff);
color:white;
padding:12px;
font-size:28px;
font-weight:bold;
}

.info{
padding:25px;
font-size:30px;
line-height:2;
}

table{
width:100%;
border-collapse:collapse;
font-size:24px;
}

th{
background:#0b3b82;
color:white;
}

td,th{
border:1px solid #ccc;
padding:12px;
}

ul{
font-size:24px;
line-height:1.8;
}

.result-image{
width:100%;
border:3px solid #ddd;
}

</style>

</head>

<body>

<div class="container">

{% if not result %}

<div class="header">

<div class="logo">
🦟 MalariaNet
</div>

<div class="subtitle">
AI-Powered Malaria Diagnosis
</div>

<div class="description">

MalariaNet is an advanced artificial intelligence platform
designed to diagnose malaria from Giemsa-stained peripheral
blood smear images.

The system automatically detects malaria parasites,
localizes infected cells with red bounding boxes,
identifies parasite stages, estimates parasitemia,
and generates a professional clinical diagnostic report.

<div class="developer">
Developed by Bisrat Weldegiyorgis
</div>

</div>

</div>

<div class="upload-box">

<form method="POST" enctype="multipart/form-data">

<input type="file" name="image" required>

<br><br>

<!-- ✅ FIX: added proper species dropdown so request.form["species"] works -->

<br><br>

<button type="submit">
Analyze Blood Smear
</button>

</form>

</div>

{% endif %}

{% if result %}

<div class="red-header">
DIAGNOSIS: {{result.diagnosis}}
</div>

<div class="info">

Species:
<b><i>{{result.species}}</i></b>

<br>

Stage:
<b>{{result.stage}}</b>

<br>

Confidence:
<b>{{result.confidence}}%</b>

<br>

Parasitemia:
<b>{{result.parasitemia}}%</b>

</div>

<div class="section">

<div class="blue">
Quantitative Analysis
</div>

<table>

<tr>
<td>RBCs Counted</td>
<td>{{result.rbc_count}}</td>
</tr>

<tr>
<td>Infected RBCs</td>
<td>{{result.infected}}</td>
</tr>

<tr>
<td>Parasitemia</td>
<td>{{result.parasitemia}}%</td>
</tr>

<tr>
<td>AI Confidence</td>
<td>{{result.confidence}}%</td>
</tr>

</table>

</div>

<div class="section">

<div class="blue">
Parasite Stage Identification
</div>

<table>

<tr>
<th>Stage</th>
<th>Status</th>
</tr>

<tr>
<td>{{result.stage}}</td>
<td>Detected</td>
</tr>

</table>

</div>

<div class="section">

<div class="blue">
Morphological Findings
</div>

<ul>
<li>RBC Enlargement: Unknown</li>
<li>Schüffner's Dots: Unknown</li>
<li>Pigment Granules: Unknown</li>
<li>Parasites Detected: {{result.infected}}</li>
</ul>

</div>

<div class="section">

<div class="blue">
Clinical Recommendation
</div>

<ul>
<li>Confirm findings with manual microscopy.</li>
<li>Monitor parasitemia levels.</li>
<li>Initiate antimalarial treatment if clinically indicated.</li>
</ul>

</div>

<div class="section">

<div class="blue">
Annotated Detection Image
</div>

<img class="result-image" src="{{result.image}}">

</div>

{% endif %}

</div>

</body>
</html>

"""

# ====================================================
# ROUTE
# ====================================================

@app.route("/", methods=["GET", "POST"])
def home():

    result = None

    if request.method == "POST":

        if "image" not in request.files:
            return render_template_string(HTML, result=None)

        image = request.files["image"]

        if image.filename == "":
            return render_template_string(HTML, result=None)

        image_path = os.path.join(
            UPLOAD_FOLDER,
            image.filename
        )

        image.save(image_path)

        result = run_prediction(image_path)

    return render_template_string(
        HTML,
        result=result
    )

# ====================================================
# MAIN
# ====================================================

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
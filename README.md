## Project Name: Cats vs Dogs
Link to GitHub repository: https://github.com/mariosam23/Cats-Vs-Dogs

---

### Description
&emsp; A web application that uses a custom AI model to classify images as either cats or dogs.
The user can upload an image and the application will return the classification result.


---

### Technologies Used
- Language → Python 3.12
- Frontend → HTML, CSS, JavaScript
- Backend → Flask

---

### Installation and Usage Instructions
1. Clone the Repository
2. Install Dependencies
   - `make build`
3. Run the Application
    - `make run`
4. Access the Web Interface
5. Show the statistis of the model, compared to a model trained on the same dataset, but with a different architecture
   - `make stats`

---

## Team Contributions
- Mario Sampetru: built the data extraction and preprocessing pipeline and the web interface
- Gabriel Carauleanu: built the AI model and the flask server, the makefile

---

### Notes
- The model was trained on a dataset of 12,500 images of cats and dogs

---

### Challenges and Solutions
- After every member of team implemented their part, we met to create an entry-point
for the project. Some issues were encountered due to the translation of mathematical
formulas using numpy. We managed to resolve them by referring to the numpy documentation.
- Another challenge was training the model, as it required plenty of time and resources.
<br>

<h5> &copy; 2023 Mario Sampetru & Gabriel Carauleanu </h5>

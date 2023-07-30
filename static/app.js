document.addEventListener("DOMContentLoaded", function () {
    const predictionForm = document.getElementById("prediction-form");
    const predictionResult = document.getElementById("prediction-result");

    predictionForm.addEventListener("submit", function (event) {
        event.preventDefault();

        const formData = new FormData(predictionForm);

        axios.post("/predict", formData)
            .then(function (response) {
                const finalResult = response.data.final_result;
                predictionResult.innerText = finalResult;
            })
            .catch(function (error) {
                console.error(error);
                predictionResult.innerText = "Error occurred while making the prediction.";
            });
    });

    // Fetch unique locations, restaurant types, and cuisines for the select boxes
    axios.get("/get_unique_values")
        .then(function (response) {
            const uniqueLocations = response.data.unique_locations;
            const uniqueRestTypes = response.data.unique_rest_types;
            const uniqueCuisines = response.data.unique_cuisines;

            const locationSelect = document.getElementById("location");
            const restTypeSelect = document.getElementById("rest_type");
            const cuisinesSelect = document.getElementById("cuisines");

            uniqueLocations.forEach((location) => {
                const option = document.createElement("option");
                option.value = location;
                option.textContent = location;
                locationSelect.appendChild(option);
            });

            uniqueRestTypes.forEach((restType) => {
                const option = document.createElement("option");
                option.value = restType;
                option.textContent = restType;
                restTypeSelect.appendChild(option);
            });

            uniqueCuisines.forEach((cuisine) => {
                const option = document.createElement("option");
                option.value = cuisine;
                option.textContent = cuisine;
                cuisinesSelect.appendChild(option);
            });
        })
        .catch(function (error) {
            console.error(error);
        });
});

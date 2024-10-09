document.getElementById('sustainability-form').addEventListener('submit', function(event) {
    event.preventDefault();  // Prevent the default form submission

    // Collect user input
    const userInput = {
        ParticipantID: parseInt(document.getElementById('participant-id').value), // Add participant ID
        Age: parseInt(document.getElementById('age').value),
        HomeSize: parseInt(document.getElementById('home-size').value),
        MonthlyElectricityConsumption: parseInt(document.getElementById('monthly-electricity').value),
        MonthlyWaterConsumption: parseInt(document.getElementById('monthly-water').value),
        Location: document.getElementById('location').value, // Use getElementById for select
        DietType: document.getElementById('diet-type').value, // Use getElementById for select
        TransportationMode: document.getElementById('transportation-mode').value, // Use getElementById for select
        EnergySource: document.getElementById('energy-source').value, // Use getElementById for select
        HomeType: document.getElementById('home-type').value, // Use getElementById for select
        Gender: document.getElementById('gender').value, // Use getElementById for select
        LocalFoodFrequency: document.getElementById('local-food-frequency').value, // Use getElementById for select
        ClothingFrequency: document.getElementById('clothing-frequency').value, // Use getElementById for select
        SustainableBrands: document.getElementById('sustainable-brands').value, // Use getElementById for select
        EnvironmentalAwareness: parseInt(document.getElementById('environmental-awareness').value), // Use getElementById for input
        CommunityInvolvement: document.getElementById('community-involvement').value, // Use getElementById for select
        UsingPlasticProducts: document.getElementById('using-plastic').value, // Use getElementById for select
        DisposalMethods: document.getElementById('disposal-methods').value, // Use getElementById for select
        PhysicalActivities: document.getElementById('physical-activities').value, // Use getElementById for select
    };

    console.log(userInput);  // Log user input to check values

    // Send data to the Flask backend
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(userInput)
    })
    .then(response => response.json())  // Convert response to JSON
    .then(data => {
        console.log('Data from server:', data);  // Log the server response
        if (data.prediction) {
            document.getElementById('result').innerText = `Sustainability Rating: ${data.prediction}`;
        } else if (data.error) {
            document.getElementById('result').innerText = `Error: ${data.error}`;
        }
    })
    .catch(error => {
        console.error('Error:', error);  // Log any errors
    });
});

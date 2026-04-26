function predictSentiment() {
    let text = document.getElementById("text").value;

    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded'
        },
        body: 'text=' + text
    })
    .then(response => response.text())
    .then(data => {
        document.getElementById("result").innerHTML = "Result: " + data;
    });
}
// source: https://codepen.io/rebelchris/pen/wvGbEVQ

var ctx = null;
var canvas = null;
var trainBtn = null;
let coord = { x: 0, y: 0 };

window.onload = () => {
    canvas = document.getElementById("canvas");
    trainBtn = document.getElementById("train-btn");
    ctx = canvas.getContext("2d");

    //start
    canvas.addEventListener("mousedown", (event) => {
        canvas.addEventListener("mousemove", draw);
        reposition(event);
    });

    //stop
    canvas.addEventListener("mouseup", () => {
        sendDrawing();
    })
    window.addEventListener("mouseup", () => {
        canvas.removeEventListener("mousemove", draw);
    });

    trainBtn.addEventListener("click", train)
    document.getElementById("reset-canvas").addEventListener("click", () => {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    })

}
function reposition(event) {
    coord.x = canvas.width * event.offsetX / canvas.clientWidth;
    coord.y = canvas.height * event.offsetY / canvas.clientHeight;
}

function draw(event) {
    ctx.beginPath();
    ctx.lineWidth = 3;
    ctx.lineCap = "round";
    ctx.strokeStyle = "#FFFFFF";
    ctx.moveTo(coord.x, coord.y);
    reposition(event);
    ctx.lineTo(coord.x, coord.y);
    ctx.stroke();
}

async function sendDrawing() {
    var pixels = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    var grayscale_data = [];
    for (let i = 0; i < pixels.length; i += 4) {
        grayscale_data.push((pixels[i] + pixels[i + 1] + pixels[i + 2]) / 3);
    }

    let response = await fetch("network/test", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        body: JSON.stringify(grayscale_data)
    });
    const jsonData = await response.json();

    let content = "Prediction:"
    for (let i = 0; i < jsonData.length; i++) {
        content += "<div><div class='legend'>" + i + "</div><div class='bar'><div style=\"width: " + Math.round(jsonData[i] * 100) + "%;\">&nbsp;</div></div><div class='percentage'>" + Math.round(jsonData[i] * 100) + "%</div></div>"

    }
    document.getElementById("predictions").innerHTML = content
    console.log(jsonData);
}

async function train(event) {
    if (trainBtn.disabled) {
        event.setCancelled(true)
    }
    trainBtn.disabled = true;
    trainBtn.value = "IN PROGRESS ...";

    let batchSize = document.getElementById("batch-size").value
    let alpha = document.getElementById("alpha").value
    let response = await fetch("network/train", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
            "Accept": "*",
        },
        body: JSON.stringify({ "batchsize": Math.round(batchSize), "alpha": Math.round(alpha * 100) / 100 })
    });
    data = await response.text();

    trainBtn.disabled = false;
    trainBtn.value = "TRAIN!"
    document.getElementById("training-div").innerText = "Training completed! Accuracy achieved: " + data
    console.log(data);
}

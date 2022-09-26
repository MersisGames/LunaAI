function encriptar() {
        var texto = document.getElementById("inputTexto").value.toLowerCase();
        var txtCifrado = texto.replace(/e/igm,"enter");
        var txtCifrado = txtCifrado.replace(/o/igm,"ober");
        var txtCifrado = txtCifrado.replace(/i/igm,"imes");
        var txtCifrado = txtCifrado.replace(/a/igm,"ai");
        var txtCifrado = txtCifrado.replace(/u/igm,"ufat")
        document.getElementById("imgDer").style.display = "none";
        document.getElementById("textDer").style.display = "none";
        document.getElementById("titleDer").style.display = "none";
        document.getElementById("textResult").style.display = "show";
        document.getElementById("textResult").style.display = "inherit";
        document.getElementById("textResult").innerHTML = txtCifrado;
        document.getElementById("copy").style.display = "show";
        document.getElementById("copy").style.display = "inherit";       
}  

function desencriptar() {
        var texto = document.getElementById("inputTexto").value.toLowerCase();
        var txtCifrado = texto.replace(/enter/igm,"e");
        var txtCifrado = txtCifrado.replace(/ober/igm,"o");
        var txtCifrado = txtCifrado.replace(/imes/igm,"i");
        var txtCifrado = txtCifrado.replace(/ai/igm,"a");
        var txtCifrado = txtCifrado.replace(/ufat/igm,"u");
        document.getElementById("imgDer").style.display = "none";
        document.getElementById("textDer").style.display = "none";
        document.getElementById("titleDer").style.display = "none";
        document.getElementById("textResult").style.display = "show";
        document.getElementById("textResult").style.display = "inherit";
        document.getElementById("textResult").innerHTML = txtCifrado;
        document.getElementById("copy").style.display = "show";
        document.getElementById("copy").style.display = "inherit";    
} 
function copy(){
        var contenido = document.querySelector("#textResult");
        contenido.select();
        document.execCommand("copy");
}
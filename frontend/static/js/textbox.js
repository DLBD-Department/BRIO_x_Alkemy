var mytextbox = document.getElementById('mytext');
var mydropdown = document.getElementById('cond_var');
mydropdown.onchange = function(){
    mytextbox.value = mytextbox.value + this.value + " ";
}

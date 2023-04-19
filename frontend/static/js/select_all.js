function select_all() {
    var select = document.getElementById("cond_var");
    var options = select.options;
    var textarea = document.getElementById("mytext");
    var values = [];
    for (var i = 0; i < options.length; i++) {
    	if (options[i].value != "select_all") {
	    options[i].selected = true;
	    values.push(options[i].value);
	}
	textarea.value = values.join(" ");
}

function handle_select() {
    var select = document.getElementById("cond_var");
    var selected_option = select.options[select.selectedIndex];
    if (selected_option.value == "select_all") {
    	select_all();
    } else {
    	var textarea = document.getElementById("mytext");
	if (textarea.value == "") {
	    textarea.value = selected_option.value;
	} else {
	    textarea.value += " " + selected_option.value;
	}
    }
}

document.addEventListener("DOMContentLoaded", function() {
    var select = document.getElementById("cond_var");
    select.addEventListener("change", handle_select);
})

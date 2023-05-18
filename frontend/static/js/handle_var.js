function handle_disab() {
    $('#cond_var').selectpicker('toggle');
    var disab = document.getElementById("root_var");
    var selected_option = disab.options[disab.selectedIndex];
    var select = document.getElementById("cond_var");
    var dd = document.getElementById("bs-select-2");
    var lis = dd.firstElementChild.getElementsByTagName('li');
    for (var j = 0; j < lis.length; j++) {
        if (lis[j].firstElementChild.lastElementChild.innerText == selected_option.value) {
            console.log(lis[j]);
            lis[j].className = "disabled";
            $('li.disabled').hide();
        }
    }
}

function handle_select() {
    var select = document.getElementById("cond_var").nextElementSibling;
    var textarea = document.getElementById("mytext");
    textarea.value = select.title;
}

function sel_all() {
    var textarea = document.getElementById("mytext");
    var select = document.getElementById("cond_var");
    var options = select.options;
    var values = [];
    for (var i = 0; i < options.length; i++) {
        values.push(options[i].value);
    }
    textarea.value = values.join(" ");
}

function desel_all() {
    var textarea = document.getElementById("mytext");
    textarea.value = "";
}

document.addEventListener("DOMContentLoaded", function() {
    var disab = document.getElementById("root_var");
    disab.addEventListener("change", handle_disab);
    var select = document.getElementById("cond_var");
    var select_all = document.getElementsByClassName("bs-actionsbox")[0].firstElementChild.firstElementChild;
    select_all.addEventListener("click", sel_all);
    var deselect_all = document.getElementsByClassName("bs-actionsbox")[0].firstElementChild.children[1];
    deselect_all.addEventListener("click", desel_all);
});

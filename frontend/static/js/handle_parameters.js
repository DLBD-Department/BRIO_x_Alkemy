function handle_disab_frompred() {
    $('#root_var').selectpicker('toggle');
    var disab = document.getElementById("predictions");
    var selected_option = disab.options[disab.selectedIndex];

    var dd_root = document.getElementById("bs-select-2");
    var lis_root = dd_root.firstElementChild.getElementsByTagName('li');
    for (var j = 0; j < lis_root.length; j++) {
	if (lis_root[j].firstElementChild.lastElementChild.innerText == selected_option.value) {
            lis_root[j].className = "disabled";
            $('li.disabled').hide();
        }
        else {
            if (lis_root[j].classList.contains("disabled")) {
                $('li.disabled').removeAttr("style");
                $('li.disabled').show();
                lis_root[j].classList.remove("disabled");
            }
        }
    }

    $('#cond_var').selectpicker('toggle');
    var dd_cond = document.getElementById("bs-select-3");
    var lis_cond = dd_cond.firstElementChild.getElementsByTagName('li');
    for (var j = 0; j < lis_cond.length; j++) {
	if (lis_cond[j].firstElementChild.lastElementChild.innerText == selected_option.value) {
            lis_cond[j].className = "disabled";
            $('li.disabled').hide();
        }
        else {
            if (lis_cond[j].classList.contains("disabled")) {
                $('li.disabled').removeAttr("style");
                $('li.disabled').show();
                lis_cond[j].classList.remove("disabled");
            }
        }
    }
}

function handle_disab_fromroot() {
    $('#cond_var').selectpicker('toggle');
    var disab = document.getElementById("root_var");
    var selected_option = disab.options[disab.selectedIndex];
    var dd = document.getElementById("bs-select-3");
    var lis = dd.firstElementChild.getElementsByTagName('li');
    for (var j = 0; j < lis.length; j++) {
	if (lis[j].firstElementChild.lastElementChild.innerText == selected_option.value) {
            lis[j].className = "disabled";
            $('li.disabled').hide();
        }
        else if (lis[j].classList.contains("disabled")) {
            $('li.disabled').show();
            lis[j].classList.remove("disabled");
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
  $('[data-bs-toggle="popover"]').popover();
  var disabfrompred = document.getElementById("predictions");
  disabfrompred.addEventListener("change", handle_disab_frompred);
  var disabfromroot = document.getElementById("root_var");
  disabfromroot.addEventListener("change", handle_disab_fromroot);
  var select = document.getElementById("cond_var");
  select.addEventListener("change", handle_select);
  var select_all = document.getElementsByClassName("bs-actionsbox")[0].firstElementChild.firstElementChild;
  select_all.addEventListener("click", sel_all);
  var deselect_all = document.getElementsByClassName("bs-actionsbox")[0].firstElementChild.children[1];
  deselect_all.addEventListener("click", desel_all);
});

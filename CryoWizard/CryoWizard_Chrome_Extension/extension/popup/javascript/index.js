
// chrome.storage.local.clear(function(){});


function cryowizard_connect_parameters_save_button_onclick() {

    var cryowizard_connect_parameters_save_button_dom = document.getElementById("cryowizard_connect_parameters_save_button");
    var cryowizard_connect_parameters_address_dom = document.getElementById("cryowizard_connect_parameters_address");
    var cryowizard_connect_parameters_email_dom = document.getElementById("cryowizard_connect_parameters_email");
    var cryowizard_connect_parameters_password_dom = document.getElementById("cryowizard_connect_parameters_password");

    cryowizard_connect_parameters_save_button_dom.textContent = "Connecting...";
    cryowizard_connect_parameters_save_button_dom.disabled = true;
    cryowizard_connect_parameters_address_dom.disabled = true;
    cryowizard_connect_parameters_email_dom.disabled = true;
    cryowizard_connect_parameters_password_dom.disabled = true;

    var socket = io(cryowizard_connect_parameters_address_dom.value);

    socket.on("connect_error", function(error) {
        alert("Connction failed, please check your CryoWizard Server Address...");

        cryowizard_connect_parameters_save_button_dom.textContent = "Save";
        cryowizard_connect_parameters_save_button_dom.disabled = false;
        cryowizard_connect_parameters_address_dom.disabled = false;
        cryowizard_connect_parameters_email_dom.disabled = false;
        cryowizard_connect_parameters_password_dom.disabled = false;
    });

    socket.on("connect_timeout", function(timeout) {
        alert("Connction failed, please check your CryoWizard Server Address...");

        cryowizard_connect_parameters_save_button_dom.textContent = "Save";
        cryowizard_connect_parameters_save_button_dom.disabled = false;
        cryowizard_connect_parameters_address_dom.disabled = false;
        cryowizard_connect_parameters_email_dom.disabled = false;
        cryowizard_connect_parameters_password_dom.disabled = false;
    });

    // check cryowizard external job parameters action backinfo
    socket.on("js_check_cryowizard_user_login_action", (res) => {
        var result = res["result"];

        if (result) {
            chrome.storage.local.set({
                "server_address": cryowizard_connect_parameters_address_dom.value,
                "cryosparc_username": cryowizard_connect_parameters_email_dom.value,
                "cryosparc_password": cryowizard_connect_parameters_password_dom.value
            }, function (){
                console.log("Parameters saved!");
            });

            alert("Connction succeed!");
            setTimeout(function(){window.close();}, 100);
        }
        else {
            alert("Connction failed, please check your CryoWizard Username and Password...");
        }

        cryowizard_connect_parameters_save_button_dom.textContent = "Save";
        cryowizard_connect_parameters_save_button_dom.disabled = false;
        cryowizard_connect_parameters_address_dom.disabled = false;
        cryowizard_connect_parameters_email_dom.disabled = false;
        cryowizard_connect_parameters_password_dom.disabled = false;

    });

    socket.emit("check_cryowizard_user_login_action", cryowizard_connect_parameters_email_dom.value, cryowizard_connect_parameters_password_dom.value);
}


$(document).ready(function () {

    chrome.storage.local.get([
        "server_address",
        "cryosparc_username",
        "cryosparc_password"
    ], function (result){
        var cryowizard_connect_parameters_address_dom = document.getElementById("cryowizard_connect_parameters_address");
        var cryowizard_connect_parameters_email_dom = document.getElementById("cryowizard_connect_parameters_email");
        var cryowizard_connect_parameters_password_dom = document.getElementById("cryowizard_connect_parameters_password");

        if (result.server_address != null){
            cryowizard_connect_parameters_address_dom.value = result.server_address;
        }
        if (result.cryosparc_username != null){
            cryowizard_connect_parameters_email_dom.value = result.cryosparc_username;
        }
        if (result.cryosparc_password != null){
            cryowizard_connect_parameters_password_dom.value = result.cryosparc_password;
        }
    });

    var cryowizard_connect_parameters_save_button_dom = document.getElementById("cryowizard_connect_parameters_save_button");
    cryowizard_connect_parameters_save_button_dom.onclick = cryowizard_connect_parameters_save_button_onclick;

});
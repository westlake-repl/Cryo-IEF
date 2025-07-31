


// session storage chrome_storage_paramters initial
chrome.storage.local.get([
    "server_address",
    "cryosparc_username",
    "cryosparc_password"
], function (result){
    var parameters = {
        "server_address": result.server_address,
        "cryosparc_username": result.cryosparc_username,
        "cryosparc_password": result.cryosparc_password
    };
    SetSessionStorageJsonItem("chrome_storage_paramters", parameters);

    var chrome_storage_paramters = GetSessionStorageJsonItem("chrome_storage_paramters");
    console.log("chrome_storage_paramters:");
    console.log(chrome_storage_paramters);
});

// set temp var
SetSessionStorageJsonItem("cryowizard_ui_parameters", {});

// set socket
var chrome_storage_paramters = GetSessionStorageJsonItem("chrome_storage_paramters");
var server_address = chrome_storage_paramters["server_address"];
const socket = io(server_address);
console.log("socketio created, server_address:", server_address);


// listen the changes of chrome.storage
chrome.storage.onChanged.addListener(function (changes, areaName) {

    chrome.storage.local.get([
        "server_address",
        "cryosparc_username",
        "cryosparc_password"
    ], function (result){
        var parameters = {
            "server_address": result.server_address,
            "cryosparc_username": result.cryosparc_username,
            "cryosparc_password": result.cryosparc_password
        };
        SetSessionStorageJsonItem("chrome_storage_paramters", parameters);
    });

    console.log("chrome.storage paramters change:");
    console.log(changes);

    var chrome_storage_paramters = GetSessionStorageJsonItem("chrome_storage_paramters");
    console.log("new chrome_storage_paramters:");
    console.log(chrome_storage_paramters);

    location.reload();

});


// main func, start after users opening Builder tab
waitForBuilderMenu((builder_elements) => {

    var chrome_storage_paramters = GetSessionStorageJsonItem("chrome_storage_paramters");
    var cryosparc_username = chrome_storage_paramters["cryosparc_username"];
    var cryosparc_password = chrome_storage_paramters["cryosparc_password"];

    // insert cryowizard buttons
    var elements = null;
    while (true) {
        elements = document.getElementsByClassName("custom-scrollbar");
        if (elements.length > 0) {
            break
        }
    }
    console.log(".custom-scrollbar found, list length:", elements.length);
    var target_element = elements[0].firstElementChild.firstElementChild;

    var new_li_string = `
    <li class="z-10 sticky top-0">
        <button type="button" class="w-full flex items-center justify-between border-t border-gray-300 bg-gray-100 px-3 py-1 text-xs font-medium text-gray-800 focus:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-offset-2 focus-visible:ring-blue-500">
            <svg xmlns="http://www.w3.org/2000/svg" class="inline-block w-4 h-4 text-gray-400" viewBox="0 0 16 16" fill="currentColor">
                <path fill-rule="evenodd" d="M2 7.75A.75.75 0 012.75 7h10a.75.75 0 010 1.5h-10A.75.75 0 012 7.75z"></path>
            </svg>
            <h3 class="font-bold">CryoWizard</h3>
            <span class="inline-flex text-gray-700">1</span>
        </button>
    </li>`;

    var new_li_2_string = `
    <li id="job_builder_cryowizard" class="border-t border-gray-300" role="menuitem">
        <div class="flex">
            <button type="button" class="ring-1 ring-inset ring-offset-2 ring-blue-500 bg-blue-100 text-blue-900 hover:bg-blue-300 w-full flex items-center justify-between px-2 py-1.5 text-left">
                <p class="text-sm">CryoWizard</p>
                <div class="flex flex-row-reverse items-center gap-1">
                    <span class="inline-flex items-center px-1 py-0.5 rounded-sm text-2xs leading-none font-medium bg-white border border-teal-400 text-teal-800">CryoWizard</span>
                </div>
            </button>
        </div>
    </li>`;

    const parser = new DOMParser();
    var new_li_doc = parser.parseFromString(new_li_string, "text/html");
    var new_li_2_doc = parser.parseFromString(new_li_2_string, "text/html");
    var new_li = new_li_doc.body.firstElementChild;
    var new_li_2 = new_li_2_doc.body.firstElementChild;

    new_li_2.firstElementChild.firstElementChild.onclick = (function (parameters) {
        return function (event) {
            if (checkURL())
            {
                var project_and_workspace = ParseProjectAndWorkspace();
                var project = project_and_workspace["project"];
                var workspace = project_and_workspace["workspace"];
                var cryosparc_username = parameters["cryosparc_username"];
                var cryosparc_password = parameters["cryosparc_password"];
                var external_default_parameters = {
                    "symmetry": "C1",
                    "diameter": "100",
                    "pixelsize": "1.0",
                    "gpu_num": "1"
                }
                CreateCryowizardExternalJobAction(cryosparc_username, cryosparc_password, project, workspace, external_default_parameters);
            }
        };
    })({
        "cryosparc_username": cryosparc_username,
        "cryosparc_password": cryosparc_password,
    });

    target_element.insertBefore(new_li_2, target_element.firstElementChild);
    target_element.insertBefore(new_li, target_element.firstElementChild);

});


// catch cryowizard building tab
waitForCryoWizardBuildingMenu((building_elements) => {

    var chrome_storage_paramters = GetSessionStorageJsonItem("chrome_storage_paramters");
    var cryosparc_username = chrome_storage_paramters["cryosparc_username"];
    var cryosparc_password = chrome_storage_paramters["cryosparc_password"];

    var project_and_job_string = building_elements.firstElementChild.firstElementChild.firstElementChild.firstElementChild.firstElementChild.textContent;
    var project = project_and_job_string.split(" ")[0].trim();
    var jobid = project_and_job_string.split(" ")[1].trim();

    var new_parameters_bar_string = `
        <button type="button" class="sticky top-0 z-11 flex-shrink-0 h-9 flex justify-between items-center font-medium px-3 bg-gray-100 hover:bg-gray-200 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-3 focus-visible:ring-offset-gray-100 focus-visible:ring-inset">
            <h2 class="font-medium">Parameters</h2>
            <div class="inline-flex items-center space-x-2">
                <span slot="badge" class="flex space-x-1"></span>
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor" class="heroicon heroicon--outline h-4 w-4 stroke-2 text-gray-600" width="100%" height="100%">
                    <path stroke-linecap="round" stroke-linejoin="round" d="M5 15l7-7 7 7"></path>
                </svg>
            </div>
        </button>`;

    var new_parameters_string = `
        <div class="bg-white p-2">
            <div class="flex flex-col">
                <div class="section expanded">
                    <span class="sticky top-9 z-2 flex flex-wrap w-full">
                        <div class="w-full h-1 bg-white flex-shrink-0"></div>
                        <button type="button" class="bg-gray-100 hover:bg-gray-200 rounded-md border border-gray-300 flex w-full items-center justify-between py-1 px-1 text-left text-sm leading-4 font-medium text-gray-800 hover:text-gray-900 focus:outline-none focus-visible:ring-inset focus-visible:ring-offset-1 focus-visible:ring-2 focus-visible:ring-blue-500">
                            <span class="flex items-center space-x-1">
                                <svg class="h-4 w-4 text-gray-500 group-hover:text-gray-500 group-focus:text-gray-600" stroke="none" fill="currentColor" viewBox="0 0 16 16">
                                    <path d="M4.42678 9.57322L7.82326 6.17678C7.92089 6.07915 8.07918 6.07915 8.17681 6.17678L11.5732 9.57322C11.7307 9.73072 11.6192 10 11.3964 10L4.60356 10C4.38083 10 4.26929 9.73071 4.42678 9.57322Z"></path>
                                </svg>
                                <span slot="header">
                                    <p class="text-sm text-gray-600">Parameters</p>
                                </span>
                            </span>
                            <span slot="badge" class="flex space-x-1">
                                <span class="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-gray-100 text-gray-800 tabular-nums"></span>
                            </span>
                        </button>
                    </span>
                    <div class="bg-white p-2">
                        <div slot="content" class="flex flex-col space-y-3 pb-2">
        
                            <div class="group">
                                <div class="flex justify-between">
                                    <label for="P34_J1053_refine_symmetry" class="block text-xs font-medium text-gray-700">Symmetry</label>
                                </div>
                                <div class="mt-1 flex items-center">
                                    <input type="text" name="` + project + "_" + jobid + `_cryowizard_symmetry" id="` + project + "_" + jobid + `_cryowizard_symmetry" class="mr-0.5 block w-full pl-2 py-1 text-xs rounded border-gray-300 focus:ring-blue-500 focus:border-blue-500" placeholder="Not set" title="">
                                    <div class="flex flex-shrink-0 items-center ml-auto">
                                        <button type="button" class="inline-flex items-center px-0.5 py-0.5 border border-transparent text-xs font-medium rounded-full text-gray-400 bg-white hover:bg-gray-100 hover:text-gray-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-gray-500 opacity-0 group-hover:opacity-100 focus:opacity-100 transition-opacity duration-150" data-trigger="tooltip-trigger-574">
                                            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" class="heroicon heroicon--solid h-5 w-5 " width="100%" height="100%">
                                                <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd"></path>
                                            </svg>
                                        </button>
                                    </div>
                                </div>
                            </div>
        
                            <div class="group">
                                <div class="flex justify-between">
                                    <label for="P34_J1053_prepare_window_inner_radius" class="block text-xs font-medium text-gray-700">Particle diameter (A)</label>
                                </div>
                                <div class="mt-1 flex items-center">
                                    <input type="number" name="` + project + "_" + jobid + `_cryowizard_diameter" id="` + project + "_" + jobid + `_cryowizard_diameter" class="mr-0.5 block w-full pl-2 py-1 text-xs rounded border-gray-300 focus:ring-blue-500 focus:border-blue-500" placeholder="Not set" title="">
                                    <div class="flex flex-shrink-0 items-center ml-auto">
                                        <button type="button" class="inline-flex items-center px-0.5 py-0.5 border border-transparent text-xs font-medium rounded-full text-gray-400 bg-white hover:bg-gray-100 hover:text-gray-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-gray-500 opacity-0 group-hover:opacity-100 focus:opacity-100 transition-opacity duration-150">
                                            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" class="heroicon heroicon--solid h-5 w-5 " width="100%" height="100%">
                                                <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd"></path>
                                            </svg>
                                        </button>
                                    </div>
                                </div>
                            </div>
                            
                            <div class="group">
                                <div class="flex justify-between">
                                    <label for="P34_J1053_prepare_window_inner_radius" class="block text-xs font-medium text-gray-700">Pixel Size (A)</label>
                                </div>
                                <div class="mt-1 flex items-center">
                                    <input type="number" name="` + project + "_" + jobid + `_cryowizard_pixelsize" id="` + project + "_" + jobid + `_cryowizard_pixelsize" class="mr-0.5 block w-full pl-2 py-1 text-xs rounded border-gray-300 focus:ring-blue-500 focus:border-blue-500" placeholder="Not set" title="">
                                    <div class="flex flex-shrink-0 items-center ml-auto">
                                        <button type="button" class="inline-flex items-center px-0.5 py-0.5 border border-transparent text-xs font-medium rounded-full text-gray-400 bg-white hover:bg-gray-100 hover:text-gray-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-gray-500 opacity-0 group-hover:opacity-100 focus:opacity-100 transition-opacity duration-150">
                                            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" class="heroicon heroicon--solid h-5 w-5 " width="100%" height="100%">
                                                <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd"></path>
                                            </svg>
                                        </button>
                                    </div>
                                </div>
                            </div>
        
                            <div class="group">
                                <div class="flex justify-between">
                                    <label for="P34_J1053_prepare_window_outer_radius" class="block text-xs font-medium text-gray-700">Number of GPUs to parallelize</label>
                                </div>
                                <div class="mt-1 flex items-center">
                                    <input type="number" name="` + project + "_" + jobid + `_cryowizard_gpu_num" id="` + project + "_" + jobid + `_cryowizard_gpu_num" class="mr-0.5 block w-full pl-2 py-1 text-xs rounded border-gray-300 focus:ring-blue-500 focus:border-blue-500" placeholder="Not set" title="">
                                    <div class="flex flex-shrink-0 items-center ml-auto">
                                        <button type="button" class="inline-flex items-center px-0.5 py-0.5 border border-transparent text-xs font-medium rounded-full text-gray-400 bg-white hover:bg-gray-100 hover:text-gray-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-gray-500 opacity-0 group-hover:opacity-100 focus:opacity-100 transition-opacity duration-150">
                                            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" class="heroicon heroicon--solid h-5 w-5 " width="100%" height="100%">
                                                <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd"></path>
                                            </svg>
                                        </button>
                                    </div>
                                </div>
                            </div>
        
                        </div>
                    </div>
                </div>
            </div>
        </div>`;

    var new_queue_job_button_string = `
        <button type="button" class="inline-flex items-center justify-center px-3 py-1.5 border border-transparent shadow-sm text-sm font-medium rounded-md text-purple-800 bg-purple-100 hover:bg-purple-200 focus:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:ring-purple-500">Queue Job</button>`;



    const parser = new DOMParser();

    // modify bug parameters bar
    var building_parameters_part_element = building_elements.children[1].lastElementChild;
    var new_parameters_bar_doc = parser.parseFromString(new_parameters_bar_string, "text/html");
    var new_parameters_bar = new_parameters_bar_doc.body.firstElementChild;
    building_parameters_part_element.removeChild(building_parameters_part_element.firstElementChild);
    building_parameters_part_element.insertBefore(new_parameters_bar, building_parameters_part_element.firstElementChild);

    // add parameters input elements
    var building_parameters_element = building_elements.children[1].lastElementChild;
    var new_parameters_doc = parser.parseFromString(new_parameters_string, "text/html");
    var new_parameters = new_parameters_doc.body.firstElementChild;
    building_parameters_element.removeChild(building_parameters_element.lastElementChild);
    building_parameters_element.appendChild(new_parameters);

    // add queue job button onclick func
    var building_queue_job_button_element = building_elements.lastElementChild;
    var new_queue_job_button_doc = parser.parseFromString(new_queue_job_button_string, "text/html");
    var new_queue_job_button = new_queue_job_button_doc.body.firstElementChild;
    new_queue_job_button.onclick = (function(parameters) {
        return function(event) {
            var building_elements = parameters["building_elements"];
            var project = parameters["project"];
            var jobid = parameters["jobid"];
            var cryosparc_username = parameters["cryosparc_username"];
            var cryosparc_password = parameters["cryosparc_password"];

            var building_card_parameters = {
                "symmetry": document.getElementById(project + "_" + jobid + "_cryowizard_symmetry").value,
                "diameter": document.getElementById(project + "_" + jobid + "_cryowizard_diameter").value,
                "pixelsize": document.getElementById(project + "_" + jobid + "_cryowizard_pixelsize").value,
                "gpu_num": document.getElementById(project + "_" + jobid + "_cryowizard_gpu_num").value
            }

            // close building page
            var building_close_button_element = building_elements.firstElementChild.lastElementChild;
            for (var iteration = 0; iteration < 5; iteration++) {
                try {
                    building_close_button_element.click();
                } catch {}
            }

            // inform cryowizard server to run
            QueueCryowizardExternalJobAction(cryosparc_username, cryosparc_password, project, jobid, building_card_parameters);
        };
    })({
        "building_elements": building_elements,
        "project": project,
        "jobid": jobid,
        "cryosparc_username": cryosparc_username,
        "cryosparc_password": cryosparc_password
    });
    building_queue_job_button_element.removeChild(building_queue_job_button_element.lastElementChild);
    building_queue_job_button_element.appendChild(new_queue_job_button);

    // get ui parameters
    CheckCryowizardExternalJobParametersAction(cryosparc_username, cryosparc_password, project, jobid);

    // check linked inputs
    var CheckCryoWizardBuildingMenuInputs_Interval= setInterval((function(parameters) {
        return function(event) {
            var project = parameters["project"];
            var jobid = parameters["jobid"];
            var building_elements = CryoWizardBuildingMenu_setInterval_Use();
            if (building_elements == null) {
                clearInterval(CheckCryoWizardBuildingMenuInputs_Interval);
            }
            else {
                CheckCryoWizardBuildingMenuInputs(building_elements.children[1].children[building_elements.children[1].children.length - 2].children[1], project, jobid);
            }
        };
    })({
        "project": project,
        "jobid": jobid
    }),100);

});


// catch cryowizard details tab
waitForCryoWizardDetailsMenu((building_elements) => {

    var chrome_storage_paramters = GetSessionStorageJsonItem("chrome_storage_paramters");
    var cryosparc_username = chrome_storage_paramters["cryosparc_username"];
    var cryosparc_password = chrome_storage_paramters["cryosparc_password"];

    var project_and_job_string = building_elements.firstElementChild.firstElementChild.firstElementChild.firstElementChild.firstElementChild.textContent;
    var project = project_and_job_string.split(" ")[0].trim();
    var jobid = project_and_job_string.split(" ")[1].trim();

    var new_parameters_bar_string = `
        <button type="button" class="sticky top-0 z-11 flex-shrink-0 h-9 flex justify-between items-center font-medium px-3 bg-gray-100 hover:bg-gray-200 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-3 focus-visible:ring-offset-gray-100 focus-visible:ring-inset">
            <h2 class="font-medium">Parameters</h2>
            <div class="inline-flex items-center space-x-2">
                <span slot="badge" class="flex space-x-1"></span>
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor" class="heroicon heroicon--outline h-4 w-4 stroke-2 text-gray-600" width="100%" height="100%">
                    <path stroke-linecap="round" stroke-linejoin="round" d="M5 15l7-7 7 7"></path>
                </svg>
            </div>
        </button>`;

    var new_parameters_string = `
        <div class="bg-white p-2">
            <div class="flex flex-col">
                <div class="section expanded">
                    <span class="sticky top-9 z-2 flex flex-wrap w-full">
                        <div class="w-full h-1 bg-white flex-shrink-0"></div>
                        <button type="button" class="bg-gray-100 hover:bg-gray-200 rounded-md border border-gray-300 flex w-full items-center justify-between py-1 px-1 text-left text-sm leading-4 font-medium text-gray-800 hover:text-gray-900 focus:outline-none focus-visible:ring-inset focus-visible:ring-offset-1 focus-visible:ring-2 focus-visible:ring-blue-500">
                            <span class="flex items-center space-x-1">
                                <svg class="h-4 w-4 text-gray-500 group-hover:text-gray-500 group-focus:text-gray-600" stroke="none" fill="currentColor" viewBox="0 0 16 16">
                                    <path d="M4.42678 9.57322L7.82326 6.17678C7.92089 6.07915 8.07918 6.07915 8.17681 6.17678L11.5732 9.57322C11.7307 9.73072 11.6192 10 11.3964 10L4.60356 10C4.38083 10 4.26929 9.73071 4.42678 9.57322Z"></path>
                                </svg>
                                <span slot="header">
                                    <p class="text-sm text-gray-600">Parameters</p>
                                </span>
                            </span>
                            <span slot="badge" class="flex space-x-1">
                                <span class="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-gray-100 text-gray-800 tabular-nums"></span>
                            </span>
                        </button>
                    </span>
                    <div class="bg-white p-2">
                        <div slot="content" class="flex flex-col space-y-3 pb-2">
        
                            <div class="group">
                                <div class="flex justify-between">
                                    <label for="P34_J1053_refine_symmetry" class="block text-xs font-medium text-gray-700">Symmetry</label>
                                </div>
                                <div class="mt-1 flex items-center">
                                    <input type="text" name="` + project + "_" + jobid + `_cryowizard_symmetry" id="` + project + "_" + jobid + `_cryowizard_symmetry" class="mr-0.5 block w-full pl-2 py-1 text-xs rounded border-gray-300 focus:ring-blue-500 focus:border-blue-500" placeholder="Not set" title="" disabled>
                                    <div class="flex flex-shrink-0 items-center ml-auto">
                                        <button type="button" class="inline-flex items-center px-0.5 py-0.5 border border-transparent text-xs font-medium rounded-full text-gray-400 bg-white hover:bg-gray-100 hover:text-gray-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-gray-500 opacity-0 group-hover:opacity-100 focus:opacity-100 transition-opacity duration-150" data-trigger="tooltip-trigger-574">
                                            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" class="heroicon heroicon--solid h-5 w-5 " width="100%" height="100%">
                                                <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd"></path>
                                            </svg>
                                        </button>
                                    </div>
                                </div>
                            </div>
        
                            <div class="group">
                                <div class="flex justify-between">
                                    <label for="P34_J1053_prepare_window_inner_radius" class="block text-xs font-medium text-gray-700">Particle diameter (A)</label>
                                </div>
                                <div class="mt-1 flex items-center">
                                    <input type="number" name="` + project + "_" + jobid + `_cryowizard_diameter" id="` + project + "_" + jobid + `_cryowizard_diameter" class="mr-0.5 block w-full pl-2 py-1 text-xs rounded border-gray-300 focus:ring-blue-500 focus:border-blue-500" placeholder="Not set" title="" disabled>
                                    <div class="flex flex-shrink-0 items-center ml-auto">
                                        <button type="button" class="inline-flex items-center px-0.5 py-0.5 border border-transparent text-xs font-medium rounded-full text-gray-400 bg-white hover:bg-gray-100 hover:text-gray-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-gray-500 opacity-0 group-hover:opacity-100 focus:opacity-100 transition-opacity duration-150">
                                            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" class="heroicon heroicon--solid h-5 w-5 " width="100%" height="100%">
                                                <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd"></path>
                                            </svg>
                                        </button>
                                    </div>
                                </div>
                            </div>
                            
                            <div class="group">
                                <div class="flex justify-between">
                                    <label for="P34_J1053_prepare_window_inner_radius" class="block text-xs font-medium text-gray-700">Pixel Size (A)</label>
                                </div>
                                <div class="mt-1 flex items-center">
                                    <input type="number" name="` + project + "_" + jobid + `_cryowizard_pixelsize" id="` + project + "_" + jobid + `_cryowizard_pixelsize" class="mr-0.5 block w-full pl-2 py-1 text-xs rounded border-gray-300 focus:ring-blue-500 focus:border-blue-500" placeholder="Not set" title="" disabled>
                                    <div class="flex flex-shrink-0 items-center ml-auto">
                                        <button type="button" class="inline-flex items-center px-0.5 py-0.5 border border-transparent text-xs font-medium rounded-full text-gray-400 bg-white hover:bg-gray-100 hover:text-gray-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-gray-500 opacity-0 group-hover:opacity-100 focus:opacity-100 transition-opacity duration-150">
                                            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" class="heroicon heroicon--solid h-5 w-5 " width="100%" height="100%">
                                                <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd"></path>
                                            </svg>
                                        </button>
                                    </div>
                                </div>
                            </div>
        
                            <div class="group">
                                <div class="flex justify-between">
                                    <label for="P34_J1053_prepare_window_outer_radius" class="block text-xs font-medium text-gray-700">Number of GPUs to parallelize</label>
                                </div>
                                <div class="mt-1 flex items-center">
                                    <input type="number" name="` + project + "_" + jobid + `_cryowizard_gpu_num" id="` + project + "_" + jobid + `_cryowizard_gpu_num" class="mr-0.5 block w-full pl-2 py-1 text-xs rounded border-gray-300 focus:ring-blue-500 focus:border-blue-500" placeholder="Not set" title="" disabled>
                                    <div class="flex flex-shrink-0 items-center ml-auto">
                                        <button type="button" class="inline-flex items-center px-0.5 py-0.5 border border-transparent text-xs font-medium rounded-full text-gray-400 bg-white hover:bg-gray-100 hover:text-gray-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-gray-500 opacity-0 group-hover:opacity-100 focus:opacity-100 transition-opacity duration-150">
                                            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" class="heroicon heroicon--solid h-5 w-5 " width="100%" height="100%">
                                                <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd"></path>
                                            </svg>
                                        </button>
                                    </div>
                                </div>
                            </div>
        
                        </div>
                    </div>
                </div>
            </div>
        </div>`;

    const parser = new DOMParser();

    // modify bug parameters bar
    var building_parameters_part_element = building_elements.children[1].lastElementChild;
    var new_parameters_bar_doc = parser.parseFromString(new_parameters_bar_string, "text/html");
    var new_parameters_bar = new_parameters_bar_doc.body.firstElementChild;
    building_parameters_part_element.removeChild(building_parameters_part_element.firstElementChild);
    building_parameters_part_element.insertBefore(new_parameters_bar, building_parameters_part_element.firstElementChild);

    // add parameters input elements
    var building_parameters_element = building_elements.children[1].lastElementChild;
    var new_parameters_doc = parser.parseFromString(new_parameters_string, "text/html");
    var new_parameters = new_parameters_doc.body.firstElementChild;
    building_parameters_element.removeChild(building_parameters_element.lastElementChild);
    building_parameters_element.appendChild(new_parameters);

    // get ui parameters
    CheckCryowizardExternalJobParametersAction(cryosparc_username, cryosparc_password, project, jobid);

});








/******************************************* functions *******************************************/


// session storage get
function GetSessionStorageJsonItem(session_storage_key)
{
    return JSON.parse(sessionStorage.getItem(session_storage_key));
}

// session storage set
function SetSessionStorageJsonItem(session_storage_key, json_value)
{
    sessionStorage.setItem(session_storage_key, JSON.stringify(json_value));
}

// switch to Buider tab: On
function waitForBuilderMenu(callback) {
    if (checkURL())
    {
        var project_and_workspace = ParseProjectAndWorkspace();
        var project = project_and_workspace["project"];
        var workspace = project_and_workspace["workspace"];

        const observer = new MutationObserver(() => {
            var elements = document.getElementsByTagName("span");
            if (elements.length > 0) {
                for (var i = 0; i < elements.length; i++) {
                    if (elements[i].textContent.trim() == ("New Job in " + project + " " + workspace)) {
                        elements[i].textContent = "New job in " + project + " " + workspace;
                        // observer.disconnect();
                        callback(elements[i]);
                    }
                }
            }
        });
        observer.observe(document.body, {childList: true, subtree: true});
    }
}

// switch to CryoWizard building tab: On
function waitForCryoWizardBuildingMenu(callback) {
    const observer = new MutationObserver(() => {
        var elements = document.getElementsByTagName("p");
        var target_element = null;
        if (elements.length > 0) {
            for (var i=0; i<elements.length; i++) {
                try {
                    target_element = elements[i].parentElement.parentElement.parentElement.parentElement.parentElement.parentElement;
                    if (
                        (target_element.children[0].children[0].lastElementChild.textContent.trim() == "External results") &&
                        (target_element.children[1].children[target_element.children[1].children.length - 2].children[1].children[0].firstElementChild.firstElementChild.textContent.trim() == "Input Movie") &&
                        (target_element.children[1].children[target_element.children[1].children.length - 2].children[1].children[1].firstElementChild.firstElementChild.textContent.trim() == "Input Micrograph") &&
                        (target_element.children[1].children[target_element.children[1].children.length - 2].children[1].children[2].firstElementChild.firstElementChild.textContent.trim() == "Input Particle") &&
                        (target_element.lastElementChild.lastElementChild.textContent.trim() == "Queue Job")
                    ) {
                        target_element.children[1].children[target_element.children[1].children.length - 2].children[1].children[0].firstElementChild.firstElementChild.textContent = "Input Movies";
                        target_element.children[1].children[target_element.children[1].children.length - 2].children[1].children[1].firstElementChild.firstElementChild.textContent = "Input Micrographs";
                        target_element.children[1].children[target_element.children[1].children.length - 2].children[1].children[2].firstElementChild.firstElementChild.textContent = "Input Particles";
                        // observer.disconnect();
                        console.log("building");
                        callback(target_element);
                    }
                } catch {}
            }
        }
    });
    observer.observe(document.body, { childList: true, subtree: true });
}

//  Used by setInterval in waitForCryoWizardBuildingMenu()
function CryoWizardBuildingMenu_setInterval_Use() {
    var elements = document.getElementsByTagName("p");
    var target_element = null;
    if (elements.length > 0) {
        for (var i=0; i<elements.length; i++) {
            try {
                target_element = elements[i].parentElement.parentElement.parentElement.parentElement.parentElement.parentElement;
                if (
                    (target_element.children[0].children[0].lastElementChild.textContent.trim() == "External results") &&
                    (target_element.children[1].children[target_element.children[1].children.length - 2].children[1].children[0].firstElementChild.firstElementChild.textContent.trim() == "Input Movies") &&
                    (target_element.children[1].children[target_element.children[1].children.length - 2].children[1].children[1].firstElementChild.firstElementChild.textContent.trim() == "Input Micrographs") &&
                    (target_element.children[1].children[target_element.children[1].children.length - 2].children[1].children[2].firstElementChild.firstElementChild.textContent.trim() == "Input Particles") &&
                    (target_element.lastElementChild.lastElementChild.textContent.trim() == "Queue Job")
                ) {
                    console.log("building setInterval");
                    return target_element;
                }
            } catch {}
        }
    }
    return null;
}

// switch to CryoWizard details tab: On
function waitForCryoWizardDetailsMenu(callback) {
    const observer = new MutationObserver(() => {
        var elements = document.getElementsByTagName("p");
        var target_element = null;
        if (elements.length > 0) {
            for (var i=0; i<elements.length; i++) {
                try {
                    target_element = elements[i].parentElement.parentElement.parentElement.parentElement.parentElement.parentElement;
                    if (
                        (target_element.children[0].children[0].lastElementChild.textContent.trim() == "External results") &&
                        (target_element.children[1].children[target_element.children[1].children.length - 2].children[1].children[0].firstElementChild.firstElementChild.textContent.trim() == "Input Movie") &&
                        (target_element.children[1].children[target_element.children[1].children.length - 2].children[1].children[1].firstElementChild.firstElementChild.textContent.trim() == "Input Micrograph") &&
                        (target_element.children[1].children[target_element.children[1].children.length - 2].children[1].children[2].firstElementChild.firstElementChild.textContent.trim() == "Input Particle") &&
                        (!(target_element.lastElementChild.lastElementChild.textContent.trim() == "Queue Job"))
                    ) {
                        target_element.children[1].children[target_element.children[1].children.length - 2].children[1].children[0].firstElementChild.firstElementChild.textContent = "Input Movies";
                        target_element.children[1].children[target_element.children[1].children.length - 2].children[1].children[1].firstElementChild.firstElementChild.textContent = "Input Micrographs";
                        target_element.children[1].children[target_element.children[1].children.length - 2].children[1].children[2].firstElementChild.firstElementChild.textContent = "Input Particles";
                        // observer.disconnect();
                        console.log("detail");
                        callback(target_element);
                    }
                } catch {}
            }
        }
    });
    observer.observe(document.body, { childList: true, subtree: true });
}

// check building tab input type
function CheckCryoWizardBuildingMenuInputs(Inputs_element, project, jobid) {
    var input_movies_num = Number(Inputs_element.children[0].firstElementChild.lastElementChild.lastElementChild.textContent.trim());
    var input_micrographs_num = Number(Inputs_element.children[1].firstElementChild.lastElementChild.lastElementChild.textContent.trim());
    var input_particles_num = Number(Inputs_element.children[2].firstElementChild.lastElementChild.lastElementChild.textContent.trim());

    // console.log("movie:", input_movies_num, typeof input_movies_num);
    // console.log("micrograph:", input_micrographs_num, typeof input_micrographs_num);
    // console.log("particle:", input_particles_num, typeof input_particles_num);

    if ((input_movies_num == 0) && (input_micrographs_num == 0) && (input_particles_num > 0)) {

        // document.getElementById(project + "_" + jobid + "_cryowizard_symmetry").disabled = true;
        document.getElementById(project + "_" + jobid + "_cryowizard_diameter").disabled = true;
        document.getElementById(project + "_" + jobid + "_cryowizard_pixelsize").disabled = true;
        document.getElementById(project + "_" + jobid + "_cryowizard_gpu_num").disabled = true;

        // document.getElementById(project + "_" + jobid + "_cryowizard_symmetry").style.backgroundColor = "#eee";
        document.getElementById(project + "_" + jobid + "_cryowizard_diameter").style.backgroundColor = "#eee";
        document.getElementById(project + "_" + jobid + "_cryowizard_pixelsize").style.backgroundColor = "#eee";
        document.getElementById(project + "_" + jobid + "_cryowizard_gpu_num").style.backgroundColor = "#eee";

    }
    else {
        document.getElementById(project + "_" + jobid + "_cryowizard_symmetry").disabled = false;
        document.getElementById(project + "_" + jobid + "_cryowizard_diameter").disabled = false;
        document.getElementById(project + "_" + jobid + "_cryowizard_pixelsize").disabled = false;
        document.getElementById(project + "_" + jobid + "_cryowizard_gpu_num").disabled = false;

        document.getElementById(project + "_" + jobid + "_cryowizard_symmetry").style.backgroundColor = "white";
        document.getElementById(project + "_" + jobid + "_cryowizard_diameter").style.backgroundColor = "white";
        document.getElementById(project + "_" + jobid + "_cryowizard_pixelsize").style.backgroundColor = "white";
        document.getElementById(project + "_" + jobid + "_cryowizard_gpu_num").style.backgroundColor = "white";
    }
}

// check url dynamically
function checkURL()
{
    console.log("Current url:", window.location.href);

    if (window.location.protocol !== 'http:') {
        return false;
    }

    var pathsplit = window.location.pathname.split("/");
    if (pathsplit.length < 3)
    {
        return false;
    }
    var PWJsplit = pathsplit[2].split("-");
    if ((PWJsplit.length != 3) || (PWJsplit[0][0] != "P") || (PWJsplit[1][0] != "W") || (PWJsplit[2][0] != "J"))
    {
        return false;
    }

    return true;
}

// get project id and workspace id from url
function ParseProjectAndWorkspace()
{
    var pathname = window.location.pathname;

    let path_split = pathname.split("/");
    var project_and_workspace = path_split[2];

    let project_and_workspace_split = project_and_workspace.split("-");
    var project = project_and_workspace_split[0];
    var workspace = project_and_workspace_split[1];

    return {"project": project, "workspace": workspace};
}

// catch target element by textContent
function CatchTargetTextElement(element_tag_name, element_text_content)
{
    var target_element_dom = null;
    // while (true)
    for (var iteration = 0; iteration < 99999; iteration++)
    {
        var elements = document.getElementsByTagName(element_tag_name);
        if (elements.length > 0)
        {
            for (var i = 0; i < elements.length; i++)
            {
                if (elements[i].textContent.trim() == element_text_content)
                {
                    target_element_dom = elements[i];
                    break;
                }
            }
        }
        if (target_element_dom != null)
            break;
    }
    return target_element_dom;
}





/******************************************* socket io *******************************************/


// check cryowizard external job parameters action
function CheckCryowizardExternalJobParametersAction(cryosparc_username, cryosparc_password, project, jobid)
{
    socket.emit("check_cryowizard_external_job_parameters_action", cryosparc_username, cryosparc_password, project, jobid);
}
// check cryowizard external job parameters action backinfo
socket.on("js_check_cryowizard_external_job_parameters_action", (res) => {
    var project = res["project"];
    var jobid = res["jobid"];
    var parameters = res["parameters"];

    document.getElementById(project + "_" + jobid + "_cryowizard_symmetry").value = parameters["symmetry"];
    document.getElementById(project + "_" + jobid + "_cryowizard_diameter").value = parameters["diameter"];
    document.getElementById(project + "_" + jobid + "_cryowizard_pixelsize").value = parameters["pixelsize"];
    document.getElementById(project + "_" + jobid + "_cryowizard_gpu_num").value = parameters["gpu_num"];
});


// create cryowizard external job action
function CreateCryowizardExternalJobAction(cryosparc_username, cryosparc_password, project, workspace, default_parameters)
{
    socket.emit("create_cryowizard_external_job_action", cryosparc_username, cryosparc_password, project, workspace, default_parameters);
}
// create cryowizard external job action backinfo
socket.on("js_create_cryowizard_external_job_action", (res) => {
    var project = res["project"];
    var workspace = res["workspace"];
    var new_external_jobid = res["new_external_jobid"];

    // var job_card_dom =  document.getElementById(project + "-" + new_external_jobid);
    // var job_card_build_button_dom = job_card_dom.lastElementChild.firstElementChild;
    // job_card_build_button_dom.click();

});


// queue cryowizard external job action
function QueueCryowizardExternalJobAction(cryosparc_username, cryosparc_password, project, jobid, parameters)
{
    socket.emit("queue_cryowizard_external_job_action", cryosparc_username, cryosparc_password, project, jobid, parameters);
}
// queue cryowizard external job action backinfo
socket.on("js_queue_cryowizard_external_job_action", (res) => {
    var project = res["project"];
    var workspace = res["workspace"];
    var new_external_jobid = res["new_external_jobid"];
});

























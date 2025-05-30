

// ====================================== whole function ===============================================================
function GetSessionStorageJsonItem(session_storage_key)
{
    return JSON.parse(sessionStorage.getItem(session_storage_key));
}

function SetSessionStorageJsonItem(session_storage_key, json_value)
{
    sessionStorage.setItem(session_storage_key, JSON.stringify(json_value));
}



// ====================================== project card =================================================================

// project card info
if (sessionStorage.getItem("project_card_count") == null)
{
    var initial_project_card_count = 1;
    var initial_project_card_dict = {
        "Workflow_0": {
            "parameters_part_safe_flag": false,
            "results_panel_part_safe_flag": false,
            "project_path": null,
            "cryosparc_username": null,
            "cryosparc_password": null,
            "cryosparc_location_project": null,
            "cryosparc_location_workspace": null,
            "refine_symmetry": null,
            "non_disabled_part": "project_dir",
            "input_card_count": 0,
            "input_card_dict": {},
            "truncation_type": "num",
            "truncation_value": null
        }
    };

    // // input_card_dict example:
    // "input_card_dict": {
    //     "movie_0": {
    //         "input_type": "movie",
    //         "input_folder_name": null,
    //         "movies_data_path": null,
    //         "gain_reference_path": null,
    //         "raw_pixel_size": null,
    //         "accelerating_voltage": null,
    //         "spherical_aberration": null,
    //         "total_exposure_dose": null,
    //         "particle_diameter": null,
    //         "gpu_num": null
    //     },
    //     "micrograph_1": {
    //         "input_type": "micrograph",
    //         "input_folder_name": null,
    //         "micrographs_data_path": null,
    //         "pixel_size": null,
    //         "accelerating_voltage": null,
    //         "spherical_aberration": null,
    //         "total_exposure_dose": null,
    //         "particle_diameter": null,
    //         "gpu_num": null
    //     },
    //     "particle_2": {
    //         "input_type": "particle",
    //         "input_folder_name": null,
    //         "source_particle_job_uid": null
    //     }
    // }

    SetSessionStorageJsonItem("project_card_count", initial_project_card_count);
    SetSessionStorageJsonItem("project_card_dict", initial_project_card_dict);
    location.reload();
}


// create html project card elements
function CreateProjectCardElements(target_project_card_index, if_active=false, non_disabled_part="project_dir", parameters_part_safe_flag=false, results_panel_part_safe_flag=false, truncation_type="num")
{
    var active_class = "";
    if (if_active)
    {
        active_class = " active";
    }

    var disabled_value_parameters_part = "";
    var disabled_value_results_panel_part = "";
    if (non_disabled_part == "project_dir")
    {
        disabled_value_parameters_part = " disabled";
        disabled_value_results_panel_part = " disabled";
    }
    else if (non_disabled_part == "parameters")
    {
        disabled_value_parameters_part = "";
        disabled_value_results_panel_part = " disabled";
    }
    else if (non_disabled_part == "results_panel")
    {
        disabled_value_parameters_part = " disabled";
        disabled_value_results_panel_part = "";
    }
    if (!(parameters_part_safe_flag && results_panel_part_safe_flag))
    {
        disabled_value_results_panel_part = " disabled";
    }

    var disabled_value_parameters_edit_button = "";
    if (parameters_part_safe_flag)
    {
        disabled_value_parameters_edit_button = "";
    }
    else
    {
        disabled_value_parameters_edit_button = " disabled";
    }

    var button_color_choose_truncation_type_num_button = "";
    var button_color_choose_truncation_type_score_button = "";
    if (truncation_type == "num")
    {
        button_color_choose_truncation_type_num_button = " btn-primary";
        button_color_choose_truncation_type_score_button = " btn-secondary";
    }
    else if (truncation_type == "score")
    {
        button_color_choose_truncation_type_num_button = " btn-secondary";
        button_color_choose_truncation_type_score_button = " btn-primary";
    }
    else
    {
        button_color_choose_truncation_type_num_button = " btn-secondary";
        button_color_choose_truncation_type_score_button = " btn-secondary";
    }

    var project_card_dict = GetSessionStorageJsonItem("project_card_dict");

    var choose_base_path = null;
    if (project_card_dict[target_project_card_index]["project_path"] == null)
    {
        choose_base_path = "~";
    }
    else
    {
        choose_base_path = project_card_dict[target_project_card_index]["project_path"];
    }

    var new_project_card_content_nav = "" +
        "<li>" +
            "<a href=\"#" + target_project_card_index + "\" class=\"nav-link text-white" + active_class + "\" data-bs-toggle=\"tab\">" +
                "<div class=\"btn-group m-0 p-0\">" +
                    "<button type=\"button\" class=\"btn text-white\" style=\"width: 150px; text-align: left\">" + target_project_card_index + "</button>" +
                    "<button type=\"button\" class=\"btn text-white\" style=\"width: 40px\" onclick=\"DeleteProjectCard(\'" + target_project_card_index + "\');\"><i class=\"bi bi-trash\"></i></button>" +
                "</div>" +
            "</a>" +
        "</li>";
    var new_project_card_content_tab = "" +
        "<div class=\"tab-pane container" + active_class + " w-100 h-100\" id=\"" + target_project_card_index + "\">" +
            // project dir
            "<div class=\"row w-100 m-0 p-0\">" +
                "<div class=\"col mb-3 mt-3\">" +
                    "<label class=\"form-label col-form-label-sm\">CryoWizard Metadata Path</label>" +
                    "<div class=\"row w-100 m-0 p-0\">" +
                        "<div class=\"col m-0 p-0\">" +
                            "<input type=\"text\" class=\"form-control form-control-sm\" id=\"" + target_project_card_index + "_project_dir\" placeholder=\"Folder to save CryoWizard running metadata\" name=\"" + target_project_card_index + "_project_dir\" disabled>" +
                        "</div>" +
                        "<div class=\"col-md-2 btn-group m-0 p-0\">" +
                            "<button type=\"button\" class=\"btn btn-primary btn-sm\" data-bs-toggle=\"modal\" data-bs-target=\"#project_modal_" + target_project_card_index + "\" onclick=\"ProjectPathGetFolderItems(\'" + target_project_card_index + "\', \'" + choose_base_path + "\');\">Choose folder</button>" +
                        "</div>" +
                    "</div>" +
                "</div>" +
            "</div>" +
            "<br>" +
            "<hr>" +
            "<br>" +
            // project info
            "<div class=\"row w-100 m-0 p-0\">" +
                "<div class=\"col mb-3 mt-3\">" +
                    "<div class=\"mb-3\">" +
                        "<label for=\"" + target_project_card_index + "_cryosparc_username\" class=\"form-label col-form-label-sm\">CryoSPARC Username</label>" +
                        "<div class=\"m-0 p-0\">" +
                            "<input type=\"text\" class=\"form-control form-control-sm\" id=\"" + target_project_card_index + "_cryosparc_username\" placeholder=\"Enter CryoSPARC Username\" name=\"" + target_project_card_index + "_cryosparc_username\"" + disabled_value_parameters_part + ">" +
                        "</div>" +
                    "</div>" +
                    "<div class=\"mb-3\">" +
                        "<label for=\"" + target_project_card_index + "_cryosparc_password\" class=\"form-label col-form-label-sm\">CryoSPARC Password</label>" +
                        "<div class=\"m-0 p-0\">" +
                            "<input type=\"password\" class=\"form-control form-control-sm\" id=\"" + target_project_card_index + "_cryosparc_password\" placeholder=\"Enter CryoSPARC Password\" name=\"" + target_project_card_index + "_cryosparc_password\"" + disabled_value_parameters_part + ">" +
                        "</div>" +
                    "</div>" +
                    "<div class=\"mb-3\">" +
                        "<label for=\"" + target_project_card_index + "_cryosparc_workspace\" class=\"form-label col-form-label-sm\">CryoSPARC Workspace</label>" +
                        "<div class=\"m-0 p-0\">" +
                            "<div class=\"row w-100 m-0 p-0\">" +
                                "<div class=\"col m-0 p-0\">" +
                                    "<input type=\"text\" class=\"form-control form-control-sm\" id=\"" + target_project_card_index + "_cryosparc_project\" placeholder=\"Enter CryoSPARC Project (e.g. P1)\" name=\"" + target_project_card_index + "_cryosparc_project\"" + disabled_value_parameters_part + ">" +
                                "</div>" +
                                "<div class=\"col m-0 p-0\">" +
                                    "<input type=\"text\" class=\"form-control form-control-sm\" id=\"" + target_project_card_index + "_cryosparc_workspace\" placeholder=\"Enter CryoSPARC Workspace (e.g. W1)\" name=\"" + target_project_card_index + "_cryosparc_workspace\"" + disabled_value_parameters_part + ">" +
                                "</div>" +
                            "</div>" +
                        "</div>" +
                    "</div>" +
                    "<div class=\"mb-3\">" +
                        "<label for=\"" + target_project_card_index + "_symmetry\" class=\"form-label col-form-label-sm\">Symmetry</label>" +
                        "<div class=\"m-0 p-0\">" +
                            "<input type=\"text\" class=\"form-control form-control-sm\" id=\"" + target_project_card_index + "_symmetry\" placeholder=\"Enter Particle Symmetry (e.g. C1)\" name=\"" + target_project_card_index + "_symmetry\"" + disabled_value_parameters_part + ">" +
                        "</div>" +
                    "</div>" +
                    "<div class=\"mb-3\">" +
                        "<label class=\"form-label col-form-label-sm\">Edit input</label>" +
                        "<div class=\"row w-100 m-0 p-0\">" +
                            "<button type=\"button\" class=\"col btn btn-secondary btn-sm\" id=\"" + target_project_card_index + "_add_movie\" onclick=\"AddInputCard(\'" + target_project_card_index + "\', \'movie\');\"" + disabled_value_parameters_part + "><i class=\"bi bi-plus\"></i>Add movies input</button>" +
                            "<button type=\"button\" class=\"col btn btn-secondary btn-sm\" id=\"" + target_project_card_index + "_add_micrograph\" onclick=\"AddInputCard(\'" + target_project_card_index + "\', \'micrograph\');\"" + disabled_value_parameters_part + "><i class=\"bi bi-plus\"></i>Add micrographs input</button>" +
                            "<button type=\"button\" class=\"col btn btn-secondary btn-sm\" id=\"" + target_project_card_index + "_add_particle\" onclick=\"AddInputCard(\'" + target_project_card_index + "\', \'particle\');\"" + disabled_value_parameters_part + "><i class=\"bi bi-plus\"></i>Add particles input</button>" +
                        "</div>" +
                        "<div class=\"row w-100 m-0 p-3\" id=\"" + target_project_card_index + "_input_card_container\"></div>" +
                    "</div>" +
                    "<div class=\"mb-3\">" +
                        "<div class=\"row m-0 p-0\">" +
                            "<button type=\"button\" class=\"col btn btn-primary btn-sm\" id=\"" + target_project_card_index + "_edit_parameter\" onclick=\"ParametersEditButtonAction(\'" + target_project_card_index + "\');\"" + disabled_value_parameters_edit_button + ">Edit Parameters</button>" +
                            "<button type=\"button\" class=\"col btn btn-primary btn-sm\" id=\"" + target_project_card_index + "_save_parameter\" onclick=\"ParametersSaveButtonAction(\'" + target_project_card_index + "\');\"" + disabled_value_parameters_part + ">Save Parameters</button>" +
                        "</div>" +
                    "</div>" +
                "</div>" +
            "</div>" +
            "<br>" +
            "<hr>" +
            "<br>" +
            // output panel
            "<div class=\"row w-100 m-0 p-0\">" +
                "<div class=\"col mb-3 mt-3\">" +
                    "<div class=\"mb-3\">" +
                        "<label class=\"form-label col-form-label-sm\">Output Panel</label>" +
                        "<div class=\"row w-100 m-0 p-0\">" +
                            "<div class=\"col bg-dark text-white w-100 m-0 p-3\" style=\"height: 800px; overflow: auto\" id=\"" + target_project_card_index + "_output_panel\"></div>" +
                            "<div class=\"col-md-1 m-0 p-0\"></div>" +
                            "<div class=\"col-md-2 m-0 p-0\">" +
                                "<div class=\"row w-100 m-0 p-0\"><button type=\"button\" class=\"col btn btn-primary btn-sm\" id=\"" + target_project_card_index + "_result_panel_run\" onclick=\"ResultPanelRunButtonAction(\'" + target_project_card_index + "\')\"" + disabled_value_results_panel_part + ">Run</button></div>" +
                                "<br>" +
                                "<div class=\"row w-100 m-0 p-0\"><button type=\"button\" class=\"col btn btn-primary btn-sm\" id=\"" + target_project_card_index + "_result_panel_kill\" onclick=\"ResultPanelKillButtonAction(\'" + target_project_card_index + "\')\"" + disabled_value_results_panel_part + ">Kill</button></div>" +
                                "<br>" +
                                "<div class=\"row w-100 m-0 p-0\"><button type=\"button\" class=\"col btn btn-primary btn-sm\" id=\"" + target_project_card_index + "_result_panel_download_map\" onclick=\"ResultPanelDownloadMapButtonAction(\'" + target_project_card_index + "\')\"" + disabled_value_results_panel_part + ">Download Map</button></div>" +
                                "<br>" +
                                "<hr>" +
                                "<br>" +
                                "<div class=\"row w-100 m-0 p-0\">" +
                                    "<div class=\"col btn-group m-0 p-0\">" +
                                        "<button type=\"button\" class=\"btn" + button_color_choose_truncation_type_num_button + " btn-sm\" id=\"" + target_project_card_index + "_result_panel_particle_truncation_type_num\" onclick=\"ResultPanelChooseTruncationType(\'" + target_project_card_index + "\', \'num\');\"" + disabled_value_results_panel_part + ">Num</button>" +
                                        "<button type=\"button\" class=\"btn" + button_color_choose_truncation_type_score_button + " btn-sm\" id=\"" + target_project_card_index + "_result_panel_particle_truncation_type_score\" onclick=\"ResultPanelChooseTruncationType(\'" + target_project_card_index + "\', \'score\');\"" + disabled_value_results_panel_part + ">Score</button>" +
                                    "</div>" +
                                    "<div class=\"col w-100 m-0 p-0\">" +
                                        "<input type=\"text\" class=\"form-control form-control-sm\" id=\"" + target_project_card_index + "_result_panel_particle_truncation_value\" name=\"" + target_project_card_index + "_result_panel_particle_truncation_value\"" + disabled_value_results_panel_part + ">" +
                                    "</div>" +
                                "</div>" +
                                "<br>" +
                                "<div class=\"row w-100 m-0 p-0\"><button type=\"button\" class=\"col btn btn-primary btn-sm\" id=\"" + target_project_card_index + "_result_panel_particle_truncation_run\" onclick=\"ResultPanelGetParticlesButtonAction(\'" + target_project_card_index + "\')\"" + disabled_value_results_panel_part + ">Get Particles</button></div>" +
                            "</div>" +
                        "</div>" +
                    "</div>" +
                "</div>" +
            "</div>" +
        "</div>" +
        // modal
        "<div class=\"modal fade\" id=\"project_modal_" + target_project_card_index + "\">" +
            "<div class=\"modal-dialog modal-lg\">" +
                "<div class=\"modal-content\">" +
                    // modal head
                    "<div class=\"modal-header\">" +
                        "<h4 class=\"modal-title\">Choose folder</h4>" +
                        "<button type=\"button\" class=\"btn-close\" data-bs-dismiss=\"modal\"></button>" +
                    "</div>" +
                    // modal body
                    "<div class=\"modal-body\">" +
                        "<div class=\"row m-0 p-0\">" +
                            "<div class=\"col m-0 p-0\">" +
                                "<input type=\"text\" class=\"form-control form-control-sm\" id=\"project_modal_folder_name_" + target_project_card_index + "\" style=\"overflow: auto\">" +
                            "</div>" +
                            "<div class=\"col-md-1 m-0 p-0\">" +
                                "<button type=\"button\" class=\"btn btn-secondary btn-sm w-100\" onclick=\"ProjectPathInputEnterAction(\'" + target_project_card_index + "\');\"><i class=\"bi bi-search\"></i></button>" +
                            "</div>" +
                        "</div>" +
                        "<div id=\"project_modal_folder_items_" + target_project_card_index + "\"></div>" +
                    "</div>" +
                    // modal foot
                    "<div class=\"modal-footer\" id=\"project_modal_bottons_" + target_project_card_index + "\"></div>" +
                "</div>" +
            "</div>" +
        "</div>";

    return [new_project_card_content_nav, new_project_card_content_tab];
}

// update html project card info
function UpdateProjectCard(active_project_card_index = null)
{
    var project_card_dict = GetSessionStorageJsonItem("project_card_dict");
    var project_card = null;
    var final_project_card_content_nav = "";
    var final_project_card_content_tab = "";
    var project_card_elements = null;
    var corrected_active_project_card_index = null;

    if (Object.keys(project_card_dict).length > 0)
    {
        if (active_project_card_index == null)
        {
            for (var project_key in project_card_dict)
            {
                corrected_active_project_card_index = project_key;
                break;
            }
        }
        else
        {
            corrected_active_project_card_index = active_project_card_index;
        }
        for (var project_key in project_card_dict)
        {
            if (project_key == corrected_active_project_card_index)
            {
                project_card_elements = CreateProjectCardElements(project_key, true, project_card_dict[project_key]["non_disabled_part"], project_card_dict[project_key]["parameters_part_safe_flag"], project_card_dict[project_key]["results_panel_part_safe_flag"], project_card_dict[project_key]["truncation_type"]);
            }
            else
            {
                project_card_elements = CreateProjectCardElements(project_key, false, project_card_dict[project_key]["non_disabled_part"], project_card_dict[project_key]["parameters_part_safe_flag"], project_card_dict[project_key]["results_panel_part_safe_flag"], project_card_dict[project_key]["truncation_type"]);
            }
            final_project_card_content_nav += project_card_elements[0];
            final_project_card_content_tab += project_card_elements[1];
        }
    }
    document.getElementById("project_card_content_div_nav").innerHTML = final_project_card_content_nav;
    document.getElementById("project_card_content_div_tab").innerHTML = final_project_card_content_tab;

    if (Object.keys(project_card_dict).length > 0)
    {
        for (var project_key in project_card_dict)
        {
            project_card = project_card_dict[project_key];
            document.getElementById(project_key + "_project_dir").value = project_card["project_path"];
            document.getElementById(project_key + "_cryosparc_username").value = project_card["cryosparc_username"];
            document.getElementById(project_key + "_cryosparc_password").value = project_card["cryosparc_password"];
            document.getElementById(project_key + "_cryosparc_project").value = project_card["cryosparc_location_project"];
            document.getElementById(project_key + "_cryosparc_workspace").value = project_card["cryosparc_location_workspace"];
            document.getElementById(project_key + "_symmetry").value = project_card["refine_symmetry"];
            document.getElementById(project_key + "_result_panel_particle_truncation_value").value = project_card["truncation_value"];
        }
    }
}

// html add new project card
function AddProjectCard()
{
    var project_card_count = GetSessionStorageJsonItem("project_card_count");
    var project_card_dict = GetSessionStorageJsonItem("project_card_dict");

    var new_project_card = {
        "parameters_part_safe_flag": false,
        "results_panel_part_safe_flag": false,
        "project_path": null,
        "cryosparc_username": null,
        "cryosparc_password": null,
        "cryosparc_location_project": null,
        "cryosparc_location_workspace": null,
        "refine_symmetry": null,
        "non_disabled_part": "project_dir",
        "input_card_count": 0,
        "input_card_dict": {},
        "truncation_type": "num",
        "truncation_value": null
    };

    var new_project_card_index = "Workflow_" + project_card_count.toString();
    project_card_dict[new_project_card_index] = new_project_card;

    SetSessionStorageJsonItem("project_card_count", project_card_count + 1);
    SetSessionStorageJsonItem("project_card_dict", project_card_dict);
    UpdateProjectCard(new_project_card_index);
    UpdateInputCard();
}

// html delete project card
function DeleteProjectCard(target_project_card_index)
{
    var project_card_dict = GetSessionStorageJsonItem("project_card_dict");
    delete project_card_dict[target_project_card_index];
    SetSessionStorageJsonItem("project_card_dict", project_card_dict);
    UpdateProjectCard();
    UpdateInputCard();
}

// html first load get parameters
function HtmlFirstLoadGetParameters()
{
    var first_load_project_card_dict = GetSessionStorageJsonItem("project_card_dict");
    var first_load_active_info;
    if (Object.keys(first_load_project_card_dict).length > 0)
    {
        for (var first_load_project_key in first_load_project_card_dict)
        {
            first_load_active_info = document.getElementById(first_load_project_key).classList.contains("active");
            if (first_load_active_info)
            {
                socket.emit("project_dir_save_button_action", first_load_project_key, first_load_project_card_dict[first_load_project_key], true);
                break;
            }
        }
    }
}

// project path input enter action
function ProjectPathInputEnterAction(target_project_card_index)
{
    ProjectPathGetFolderItems(target_project_card_index, document.getElementById("project_modal_folder_name_" + target_project_card_index).value);
}

// project add new folder action
function ProjectAddNewFolderAction(target_project_card_index, target_folder)
{
    socket.emit("project_dir_add_new_folder_action", target_project_card_index, target_folder, document.getElementById("project_add_new_folder_name_" + target_project_card_index).value);
}

// project delete folder action
function ProjectDeleteFolderAction(target_project_card_index, target_folder, delete_folder_name)
{
    socket.emit("project_dir_delete_folder_action", target_project_card_index, target_folder, delete_folder_name);
}

// project path get folder items
function ProjectPathGetFolderItems(target_project_card_index, target_folder)
{
    socket.emit("project_dir_get_folder_items_action", target_project_card_index, target_folder);
}

// // project path edit button action
// function ProjectPathEditButtonAction(target_project_card_index)
// {
//     var project_card_dict = GetSessionStorageJsonItem("project_card_dict");
//
//     project_card_dict[target_project_card_index] = SaveWrittenParameters(project_card_dict[target_project_card_index], target_project_card_index);
//
//     project_card_dict[target_project_card_index]["parameters_part_safe_flag"] = false;
//     project_card_dict[target_project_card_index]["results_panel_part_safe_flag"] = false;
//     project_card_dict[target_project_card_index]["non_disabled_part"] = "project_dir";
//
//     SetSessionStorageJsonItem("project_card_dict", project_card_dict);
//     UpdateProjectCard(target_project_card_index);
//     UpdateInputCard();
// }

// project path save button action
function ProjectPathSaveButtonAction(target_project_card_index, chose_project_path)
{
    var project_card_dict = GetSessionStorageJsonItem("project_card_dict");

    project_card_dict[target_project_card_index] = SaveWrittenParameters(project_card_dict[target_project_card_index], target_project_card_index);

    project_card_dict[target_project_card_index]["project_path"] = chose_project_path;
    socket.emit("project_dir_save_button_action", target_project_card_index, project_card_dict[target_project_card_index], false);
}

// parameter part edit button action
function ParametersEditButtonAction(target_project_card_index)
{
    var project_card_dict = GetSessionStorageJsonItem("project_card_dict");

    project_card_dict[target_project_card_index] = SaveWrittenParameters(project_card_dict[target_project_card_index], target_project_card_index);

    project_card_dict[target_project_card_index]["non_disabled_part"] = "parameters";
    project_card_dict[target_project_card_index]["results_panel_part_safe_flag"] = false;

    SetSessionStorageJsonItem("project_card_dict", project_card_dict);
    UpdateProjectCard(target_project_card_index);
    UpdateInputCard();
}

// parameter part save button action
function ParametersSaveButtonAction(target_project_card_index)
{
    var project_card_dict = GetSessionStorageJsonItem("project_card_dict");

    project_card_dict[target_project_card_index] = SaveWrittenParameters(project_card_dict[target_project_card_index], target_project_card_index);

    if (project_card_dict[target_project_card_index]["parameters_part_safe_flag"])
    {
        project_card_dict[target_project_card_index]["non_disabled_part"] = "results_panel";
        project_card_dict[target_project_card_index] = SaveWrittenParameters(project_card_dict[target_project_card_index], target_project_card_index);

        socket.emit("parameters_save_button_action", target_project_card_index, project_card_dict[target_project_card_index]);
    }
}



// ====================================== movie/micrograph/particle input card =========================================

// create html project card elements
function CreateInputCardElements(target_project_card_index, target_input_card_index, target_input_type, non_disabled_part)
{
    var disabled_value_parameters_part = "";
    if (non_disabled_part == "parameters")
    {
        disabled_value_parameters_part = "";
    }
    else
    {
        disabled_value_parameters_part = " disabled";
    }

    var new_input_card_content = "";
    if (target_input_type == "movie")
    {
        new_input_card_content = "" +
            "<div class=\"col-md-6 m-0 p-3 border\" style=\"height: 400px\">" +
                "<div class=\"row m-0 p-0\">" +
                    "<div class=\"col-md-1 m-0 p-0\"></div>" +
                    "<div class=\"col m-0 p-0 text-center\"><label class=\"form-label col-form-label-sm\">Import Movie</label></div>" +
                    "<div class=\"col-md-1 m-0 p-0\"><button type=\"button\" class=\"btn btn-sm text-dark\" onclick=\"DeleteInputCard(\'" + target_project_card_index + "\', \'" + target_input_card_index + "\');\"" + disabled_value_parameters_part + "><i class=\"bi bi-trash\"></i></button></div>" +
                "</div>" +
                "<div class=\"row m-0 p-0\">" +
                    "<div class=\"col m-0 p-0\"><label for=\"" + target_project_card_index + "_" + target_input_card_index + "_movies_data_path\" class=\"form-label col-form-label-sm\">Movies data path</label></div>" +
                    "<div class=\"col m-0 p-0\"><input type=\"text\" class=\"form-control form-control-sm\" id=\"" + target_project_card_index + "_" + target_input_card_index + "_movies_data_path\" name=\"" + target_project_card_index + "_" + target_input_card_index + "_movies_data_path\"" + disabled_value_parameters_part + "></div>" +
                "</div>" +
                "<div class=\"row m-0 p-0\">" +
                    "<div class=\"col m-0 p-0\"><label for=\"" + target_project_card_index + "_" + target_input_card_index + "_gain_reference_path\" class=\"form-label col-form-label-sm\">Gain reference path</label></div>" +
                    "<div class=\"col m-0 p-0\"><input type=\"text\" class=\"form-control form-control-sm\" id=\"" + target_project_card_index + "_" + target_input_card_index + "_gain_reference_path\" name=\"" + target_project_card_index + "_" + target_input_card_index + "_gain_reference_path\"" + disabled_value_parameters_part + "></div>" +
                "</div>" +
                "<div class=\"row m-0 p-0\">" +
                    "<div class=\"col m-0 p-0\"><label for=\"" + target_project_card_index + "_" + target_input_card_index + "_raw_pixel_size\" class=\"form-label col-form-label-sm\">Raw pixel size (A)</label></div>" +
                    "<div class=\"col m-0 p-0\"><input type=\"text\" class=\"form-control form-control-sm\" id=\"" + target_project_card_index + "_" + target_input_card_index + "_raw_pixel_size\" name=\"" + target_project_card_index + "_" + target_input_card_index + "_raw_pixel_size\"" + disabled_value_parameters_part + "></div>" +
                "</div>" +
                "<div class=\"row m-0 p-0\">" +
                    "<div class=\"col m-0 p-0\"><label for=\"" + target_project_card_index + "_" + target_input_card_index + "_accelerating_voltage\" class=\"form-label col-form-label-sm\">Accelerating Voltage (kV)</label></div>" +
                    "<div class=\"col m-0 p-0\"><input type=\"text\" class=\"form-control form-control-sm\" id=\"" + target_project_card_index + "_" + target_input_card_index + "_accelerating_voltage\" name=\"" + target_project_card_index + "_" + target_input_card_index + "_accelerating_voltage\"" + disabled_value_parameters_part + "></div>" +
                "</div>" +
                "<div class=\"row m-0 p-0\">" +
                    "<div class=\"col m-0 p-0\"><label for=\"" + target_project_card_index + "_" + target_input_card_index + "_spherical_aberration\" class=\"form-label col-form-label-sm\">Spherical Aberration (mm)</label></div>" +
                    "<div class=\"col m-0 p-0\"><input type=\"text\" class=\"form-control form-control-sm\" id=\"" + target_project_card_index + "_" + target_input_card_index + "_spherical_aberration\" name=\"" + target_project_card_index + "_" + target_input_card_index + "_spherical_aberration\"" + disabled_value_parameters_part + "></div>" +
                "</div>" +
                "<div class=\"row m-0 p-0\">" +
                    "<div class=\"col m-0 p-0\"><label for=\"" + target_project_card_index + "_" + target_input_card_index + "_total_exposure_dose\" class=\"form-label col-form-label-sm\">Total exposure dose (e/A^2)</label></div>" +
                    "<div class=\"col m-0 p-0\"><input type=\"text\" class=\"form-control form-control-sm\" id=\"" + target_project_card_index + "_" + target_input_card_index + "_total_exposure_dose\" name=\"" + target_project_card_index + "_" + target_input_card_index + "_total_exposure_dose\"" + disabled_value_parameters_part + "></div>" +
                "</div>" +
                "<div class=\"row m-0 p-0\">" +
                    "<div class=\"col m-0 p-0\"><label for=\"" + target_project_card_index + "_" + target_input_card_index + "_particle_diameter\" class=\"form-label col-form-label-sm\">Particle diameter (A)</label></div>" +
                    "<div class=\"col m-0 p-0\"><input type=\"text\" class=\"form-control form-control-sm\" id=\"" + target_project_card_index + "_" + target_input_card_index + "_particle_diameter\" name=\"" + target_project_card_index + "_" + target_input_card_index + "_particle_diameter\"" + disabled_value_parameters_part + "></div>" +
                "</div>" +
                "<div class=\"row m-0 p-0\">" +
                    "<div class=\"col m-0 p-0\"><label for=\"" + target_project_card_index + "_" + target_input_card_index + "_gpu_num\" class=\"form-label col-form-label-sm\">Number of GPUs to parallelize</label></div>" +
                    "<div class=\"col m-0 p-0\"><input type=\"text\" class=\"form-control form-control-sm\" id=\"" + target_project_card_index + "_" + target_input_card_index + "_gpu_num\" name=\"" + target_project_card_index + "_" + target_input_card_index + "_gpu_num\"" + disabled_value_parameters_part + "></div>" +
                "</div>" +
            "</div>";
    }
    else if (target_input_type == "micrograph")
    {
        new_input_card_content = "" +
            "<div class=\"col-md-6 m-0 p-3 border\" style=\"height: 400px\">" +
                "<div class=\"row m-0 p-0\">" +
                    "<div class=\"col-md-1 m-0 p-0\"></div>" +
                    "<div class=\"col m-0 p-0 text-center\"><label class=\"form-label col-form-label-sm\">Import Micrograph</label></div>" +
                    "<div class=\"col-md-1 m-0 p-0\"><button type=\"button\" class=\"btn btn-sm text-dark\" onclick=\"DeleteInputCard(\'" + target_project_card_index + "\', \'" + target_input_card_index + "\');\"" + disabled_value_parameters_part + "><i class=\"bi bi-trash\"></i></button></div>" +
                "</div>" +
                "<div class=\"row m-0 p-0\">" +
                    "<div class=\"col m-0 p-0\"><label for=\"" + target_project_card_index + "_" + target_input_card_index + "_micrographs_data_path\" class=\"form-label col-form-label-sm\">Micrographs data path</label></div>" +
                    "<div class=\"col m-0 p-0\"><input type=\"text\" class=\"form-control form-control-sm\" id=\"" + target_project_card_index + "_" + target_input_card_index + "_micrographs_data_path\" name=\"" + target_project_card_index + "_" + target_input_card_index + "_micrographs_data_path\"" + disabled_value_parameters_part + "></div>" +
                "</div>" +
                "<div class=\"row m-0 p-0\">" +
                    "<div class=\"col m-0 p-0\"><label for=\"" + target_project_card_index + "_" + target_input_card_index + "_pixel_size\" class=\"form-label col-form-label-sm\">Pixel size (A)</label></div>" +
                    "<div class=\"col m-0 p-0\"><input type=\"text\" class=\"form-control form-control-sm\" id=\"" + target_project_card_index + "_" + target_input_card_index + "_pixel_size\" name=\"" + target_project_card_index + "_" + target_input_card_index + "_pixel_size\"" + disabled_value_parameters_part + "></div>" +
                "</div>" +
                "<div class=\"row m-0 p-0\">" +
                    "<div class=\"col m-0 p-0\"><label for=\"" + target_project_card_index + "_" + target_input_card_index + "_accelerating_voltage\" class=\"form-label col-form-label-sm\">Accelerating Voltage (kV)</label></div>" +
                    "<div class=\"col m-0 p-0\"><input type=\"text\" class=\"form-control form-control-sm\" id=\"" + target_project_card_index + "_" + target_input_card_index + "_accelerating_voltage\" name=\"" + target_project_card_index + "_" + target_input_card_index + "_accelerating_voltage\"" + disabled_value_parameters_part + "></div>" +
                "</div>" +
                "<div class=\"row m-0 p-0\">" +
                    "<div class=\"col m-0 p-0\"><label for=\"" + target_project_card_index + "_" + target_input_card_index + "_spherical_aberration\" class=\"form-label col-form-label-sm\">Spherical Aberration (mm)</label></div>" +
                    "<div class=\"col m-0 p-0\"><input type=\"text\" class=\"form-control form-control-sm\" id=\"" + target_project_card_index + "_" + target_input_card_index + "_spherical_aberration\" name=\"" + target_project_card_index + "_" + target_input_card_index + "_spherical_aberration\"" + disabled_value_parameters_part + "></div>" +
                "</div>" +
                "<div class=\"row m-0 p-0\">" +
                    "<div class=\"col m-0 p-0\"><label for=\"" + target_project_card_index + "_" + target_input_card_index + "_total_exposure_dose\" class=\"form-label col-form-label-sm\">Total exposure dose (e/A^2)</label></div>" +
                    "<div class=\"col m-0 p-0\"><input type=\"text\" class=\"form-control form-control-sm\" id=\"" + target_project_card_index + "_" + target_input_card_index + "_total_exposure_dose\" name=\"" + target_project_card_index + "_" + target_input_card_index + "_total_exposure_dose\"" + disabled_value_parameters_part + "></div>" +
                "</div>" +
                "<div class=\"row m-0 p-0\">" +
                    "<div class=\"col m-0 p-0\"><label for=\"" + target_project_card_index + "_" + target_input_card_index + "_particle_diameter\" class=\"form-label col-form-label-sm\">Particle diameter (A)</label></div>" +
                    "<div class=\"col m-0 p-0\"><input type=\"text\" class=\"form-control form-control-sm\" id=\"" + target_project_card_index + "_" + target_input_card_index + "_particle_diameter\" name=\"" + target_project_card_index + "_" + target_input_card_index + "_particle_diameter\"" + disabled_value_parameters_part + "></div>" +
                "</div>" +
                "<div class=\"row m-0 p-0\">" +
                    "<div class=\"col m-0 p-0\"><label for=\"" + target_project_card_index + "_" + target_input_card_index + "_gpu_num\" class=\"form-label col-form-label-sm\">Number of GPUs to parallelize</label></div>" +
                    "<div class=\"col m-0 p-0\"><input type=\"text\" class=\"form-control form-control-sm\" id=\"" + target_project_card_index + "_" + target_input_card_index + "_gpu_num\" name=\"" + target_project_card_index + "_" + target_input_card_index + "_gpu_num\"" + disabled_value_parameters_part + "></div>" +
                "</div>" +
            "</div>";
    }
    else if (target_input_type == "particle")
    {
        new_input_card_content = "" +
            "<div class=\"col-md-6 m-0 p-3 border\" style=\"height: 400px\">" +
                "<div class=\"row m-0 p-0\">" +
                    "<div class=\"col-md-1 m-0 p-0\"></div>" +
                    "<div class=\"col m-0 p-0 text-center\"><label class=\"form-label col-form-label-sm\">Import Particle</label></div>" +
                    "<div class=\"col-md-1 m-0 p-0\"><button type=\"button\" class=\"btn btn-sm text-dark\" onclick=\"DeleteInputCard(\'" + target_project_card_index + "\', \'" + target_input_card_index + "\');\"" + disabled_value_parameters_part + "><i class=\"bi bi-trash\"></i></button></div>" +
                "</div>" +
                "<div class=\"row m-0 p-0\">" +
                    "<div class=\"col m-0 p-0\"><label for=\"" + target_project_card_index + "_" + target_input_card_index + "_source_particle_job_uid\" class=\"form-label col-form-label-sm\">Source Particle Job uid</label></div>" +
                    "<div class=\"col m-0 p-0\"><input type=\"text\" class=\"form-control form-control-sm\" id=\"" + target_project_card_index + "_" + target_input_card_index + "_source_particle_job_uid\" name=\"" + target_project_card_index + "_" + target_input_card_index + "_source_particle_job_uid\"" + disabled_value_parameters_part + "></div>" +
                "</div>" +
            "</div>";
    }
    else
    {
        new_input_card_content = "";
    }

    return new_input_card_content;
}

// save written parameters
function SaveWrittenParameters(target_project_card_dict, target_project_card_index)
{
    var saved_target_project_card_dict = target_project_card_dict;
    var input_card_dict = saved_target_project_card_dict["input_card_dict"];

    saved_target_project_card_dict["cryosparc_username"] = String(document.getElementById(target_project_card_index + "_cryosparc_username").value);
    saved_target_project_card_dict["cryosparc_password"] = String(document.getElementById(target_project_card_index + "_cryosparc_password").value);
    saved_target_project_card_dict["cryosparc_location_project"] = String(document.getElementById(target_project_card_index + "_cryosparc_project").value);
    saved_target_project_card_dict["cryosparc_location_workspace"] = String(document.getElementById(target_project_card_index + "_cryosparc_workspace").value);
    saved_target_project_card_dict["refine_symmetry"] = String(document.getElementById(target_project_card_index + "_symmetry").value);
    saved_target_project_card_dict["truncation_value"] = String(document.getElementById(target_project_card_index + "_result_panel_particle_truncation_value").value);
    if (Object.keys(input_card_dict).length > 0)
    {
        for (var input_key in input_card_dict)
        {
            if (input_card_dict[input_key]["input_type"] == "movie")
            {
                input_card_dict[input_key]["movies_data_path"] = String(document.getElementById(target_project_card_index + "_" + input_key + "_movies_data_path").value);
                input_card_dict[input_key]["gain_reference_path"] = String(document.getElementById(target_project_card_index + "_" + input_key + "_gain_reference_path").value);
                input_card_dict[input_key]["raw_pixel_size"] = String(document.getElementById(target_project_card_index + "_" + input_key + "_raw_pixel_size").value);
                input_card_dict[input_key]["accelerating_voltage"] = String(document.getElementById(target_project_card_index + "_" + input_key + "_accelerating_voltage").value);
                input_card_dict[input_key]["spherical_aberration"] = String(document.getElementById(target_project_card_index + "_" + input_key + "_spherical_aberration").value);
                input_card_dict[input_key]["total_exposure_dose"] = String(document.getElementById(target_project_card_index + "_" + input_key + "_total_exposure_dose").value);
                input_card_dict[input_key]["particle_diameter"] = String(document.getElementById(target_project_card_index + "_" + input_key + "_particle_diameter").value);
                input_card_dict[input_key]["gpu_num"] = String(document.getElementById(target_project_card_index + "_" + input_key + "_gpu_num").value);
            }
            else if (input_card_dict[input_key]["input_type"] == "micrograph")
            {
                input_card_dict[input_key]["micrographs_data_path"] = String(document.getElementById(target_project_card_index + "_" + input_key + "_micrographs_data_path").value);
                input_card_dict[input_key]["pixel_size"] = String(document.getElementById(target_project_card_index + "_" + input_key + "_pixel_size").value);
                input_card_dict[input_key]["accelerating_voltage"] = String(document.getElementById(target_project_card_index + "_" + input_key + "_accelerating_voltage").value);
                input_card_dict[input_key]["spherical_aberration"] = String(document.getElementById(target_project_card_index + "_" + input_key + "_spherical_aberration").value);
                input_card_dict[input_key]["total_exposure_dose"] = String(document.getElementById(target_project_card_index + "_" + input_key + "_total_exposure_dose").value);
                input_card_dict[input_key]["particle_diameter"] = String(document.getElementById(target_project_card_index + "_" + input_key + "_particle_diameter").value);
                input_card_dict[input_key]["gpu_num"] = String(document.getElementById(target_project_card_index + "_" + input_key + "_gpu_num").value);
            }
            else if (input_card_dict[input_key]["input_type"] == "particle")
            {
                input_card_dict[input_key]["source_particle_job_uid"] = String(document.getElementById(target_project_card_index + "_" + input_key + "_source_particle_job_uid").value);
            }
        }
    }
    saved_target_project_card_dict["input_card_dict"] = input_card_dict;

    return saved_target_project_card_dict;
}

// update html input card info
function UpdateInputCard()
{
    var project_card_dict = GetSessionStorageJsonItem("project_card_dict");
    var final_input_card_content = "";
    var input_card_dict = null;
    var input_card_elements = null;

    if (Object.keys(project_card_dict).length > 0)
    {
        for (var project_key in project_card_dict)
        {
            final_input_card_content = "";
            input_card_dict = project_card_dict[project_key]["input_card_dict"];

            if (Object.keys(input_card_dict).length > 0)
            {
                for (var input_key in input_card_dict)
                {
                    input_card_elements = CreateInputCardElements(project_key, input_key, input_card_dict[input_key]["input_type"], project_card_dict[project_key]["non_disabled_part"]);
                    final_input_card_content += input_card_elements;
                }
            }
            document.getElementById(project_key + "_input_card_container").innerHTML = final_input_card_content;

            if (Object.keys(input_card_dict).length > 0)
            {
                for (var input_key in input_card_dict)
                {
                    if (input_card_dict[input_key]["input_type"] == "movie")
                    {
                        document.getElementById(project_key + "_" + input_key + "_movies_data_path").value = input_card_dict[input_key]["movies_data_path"];
                        document.getElementById(project_key + "_" + input_key + "_gain_reference_path").value = input_card_dict[input_key]["gain_reference_path"];
                        document.getElementById(project_key + "_" + input_key + "_raw_pixel_size").value = input_card_dict[input_key]["raw_pixel_size"];
                        document.getElementById(project_key + "_" + input_key + "_accelerating_voltage").value = input_card_dict[input_key]["accelerating_voltage"];
                        document.getElementById(project_key + "_" + input_key + "_spherical_aberration").value = input_card_dict[input_key]["spherical_aberration"];
                        document.getElementById(project_key + "_" + input_key + "_total_exposure_dose").value = input_card_dict[input_key]["total_exposure_dose"];
                        document.getElementById(project_key + "_" + input_key + "_particle_diameter").value = input_card_dict[input_key]["particle_diameter"];
                        document.getElementById(project_key + "_" + input_key + "_gpu_num").value = input_card_dict[input_key]["gpu_num"];
                    }
                    else if (input_card_dict[input_key]["input_type"] == "micrograph")
                    {
                        document.getElementById(project_key + "_" + input_key + "_micrographs_data_path").value = input_card_dict[input_key]["micrographs_data_path"];
                        document.getElementById(project_key + "_" + input_key + "_pixel_size").value = input_card_dict[input_key]["pixel_size"];
                        document.getElementById(project_key + "_" + input_key + "_accelerating_voltage").value = input_card_dict[input_key]["accelerating_voltage"];
                        document.getElementById(project_key + "_" + input_key + "_spherical_aberration").value = input_card_dict[input_key]["spherical_aberration"];
                        document.getElementById(project_key + "_" + input_key + "_total_exposure_dose").value = input_card_dict[input_key]["total_exposure_dose"];
                        document.getElementById(project_key + "_" + input_key + "_particle_diameter").value = input_card_dict[input_key]["particle_diameter"];
                        document.getElementById(project_key + "_" + input_key + "_gpu_num").value = input_card_dict[input_key]["gpu_num"];
                    }
                    else if (input_card_dict[input_key]["input_type"] == "particle")
                    {
                        document.getElementById(project_key + "_" + input_key + "_source_particle_job_uid").value = input_card_dict[input_key]["source_particle_job_uid"];
                    }
                }
            }
        }
    }
}

// html add new input card
function AddInputCard(target_project_card_index, target_input_type)
{
    var project_card_dict = GetSessionStorageJsonItem("project_card_dict");
    var input_card_count = project_card_dict[target_project_card_index]["input_card_count"];
    var input_card_dict = project_card_dict[target_project_card_index]["input_card_dict"];

    project_card_dict[target_project_card_index] = SaveWrittenParameters(project_card_dict[target_project_card_index], target_project_card_index);

    // create new input card
    var new_input_card_index = target_input_type + "_" + input_card_count.toString();
    if (target_input_type == "movie")
    {
        input_card_dict[new_input_card_index] = {
            "input_type": "movie",
            "input_folder_name": null,
            "movies_data_path": null,
            "gain_reference_path": null,
            "raw_pixel_size": null,
            "accelerating_voltage": null,
            "spherical_aberration": null,
            "total_exposure_dose": null,
            "particle_diameter": null,
            "gpu_num": null
        };
        project_card_dict[target_project_card_index]["input_card_dict"] = input_card_dict;
        project_card_dict[target_project_card_index]["input_card_count"] = input_card_count + 1;
    }
    else if (target_input_type == "micrograph")
    {
        input_card_dict[new_input_card_index] = {
            "input_type": "micrograph",
            "input_folder_name": null,
            "micrographs_data_path": null,
            "pixel_size": null,
            "accelerating_voltage": null,
            "spherical_aberration": null,
            "total_exposure_dose": null,
            "particle_diameter": null,
            "gpu_num": null
        };
        project_card_dict[target_project_card_index]["input_card_dict"] = input_card_dict;
        project_card_dict[target_project_card_index]["input_card_count"] = input_card_count + 1;
    }
    else if (target_input_type == "particle")
    {
        input_card_dict[new_input_card_index] = {
            "input_type": "particle",
            "input_folder_name": null,
            "source_particle_job_uid": null
        };
        project_card_dict[target_project_card_index]["input_card_dict"] = input_card_dict;
        project_card_dict[target_project_card_index]["input_card_count"] = input_card_count + 1;
    }

    SetSessionStorageJsonItem("project_card_dict", project_card_dict);
    UpdateProjectCard(target_project_card_index);
    UpdateInputCard();
}

// html delete input card
function DeleteInputCard(target_project_card_index, target_input_card_index)
{
    var project_card_dict = GetSessionStorageJsonItem("project_card_dict");
    var input_card_dict = project_card_dict[target_project_card_index]["input_card_dict"];
    delete input_card_dict[target_input_card_index];
    project_card_dict[target_project_card_index]["input_card_dict"] = input_card_dict;
    SetSessionStorageJsonItem("project_card_dict", project_card_dict);
    UpdateProjectCard(target_project_card_index);
    UpdateInputCard();
}



// ====================================== result panel =================================================================

// result panel choose truncation type
function ResultPanelChooseTruncationType(target_project_card_index, truncation_type="num")
{
    var project_card_dict = GetSessionStorageJsonItem("project_card_dict");

    project_card_dict[target_project_card_index] = SaveWrittenParameters(project_card_dict[target_project_card_index], target_project_card_index);

    if (truncation_type == "num")
    {
        project_card_dict[target_project_card_index]["truncation_type"] = "num";
    }
    else if (truncation_type == "score")
    {
        project_card_dict[target_project_card_index]["truncation_type"] = "score";
    }

    SetSessionStorageJsonItem("project_card_dict", project_card_dict);
    UpdateProjectCard(target_project_card_index);
    UpdateInputCard();
}

// result panel run button action
function ResultPanelRunButtonAction(target_project_card_index)
{
    var project_card_dict = GetSessionStorageJsonItem("project_card_dict");

    if (project_card_dict[target_project_card_index]["parameters_part_safe_flag"])
    {
        socket.emit("result_panel_run_button_action", target_project_card_index, project_card_dict[target_project_card_index]["project_path"]);
    }
}

// result panel run button action
function ResultPanelKillButtonAction(target_project_card_index)
{
    var project_card_dict = GetSessionStorageJsonItem("project_card_dict");

    if (project_card_dict[target_project_card_index]["parameters_part_safe_flag"])
    {
        socket.emit("result_panel_kill_button_action", target_project_card_index, project_card_dict[target_project_card_index]["project_path"]);
    }
}

// result panel download map button action
function ResultPanelDownloadMapButtonAction(target_project_card_index)
{
    var project_card_dict = GetSessionStorageJsonItem("project_card_dict");

    if (project_card_dict[target_project_card_index]["parameters_part_safe_flag"])
    {
        var download_map_url = window.location.href + "/DownloadMap?project_dir=" + project_card_dict[target_project_card_index]["project_path"] + "&project_card=" + target_project_card_index;
        window.open(download_map_url);
    }
}

// result panel get particles button action
function ResultPanelGetParticlesButtonAction(target_project_card_index)
{
    var project_card_dict = GetSessionStorageJsonItem("project_card_dict");

    if (project_card_dict[target_project_card_index]["parameters_part_safe_flag"])
    {
        socket.emit("result_panel_get_particles_button_action", target_project_card_index, project_card_dict[target_project_card_index]["project_path"], project_card_dict[target_project_card_index]["truncation_type"], document.getElementById(target_project_card_index + "_result_panel_particle_truncation_value").value);
    }
}

// choose active project to show output panel results
function ChooseActiveProjectToShowResults()
{
    var project_card_dict = GetSessionStorageJsonItem("project_card_dict");

    var active_info;
    if (Object.keys(project_card_dict).length > 0)
    {
        for (var project_key in project_card_dict)
        {
            active_info = document.getElementById(project_key).classList.contains("active");
            if (active_info)
            {
                socket.emit("result_panel_output_panel_show_data", project_key, project_card_dict[project_key]["project_path"]);
                break;
            }
        }
    }
}
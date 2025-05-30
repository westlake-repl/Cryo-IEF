

// project path get folder items, handle server response
socket.on("js_project_dir_get_folder_items_action", (res) => {
    var single_project_card_index = res["single_project_card_index"];
    var response_folder = res["response_folder"];
    var response_folder_items = res["response_folder_items"];
    var response_parent_folder = res["response_parent_folder"];

    var project_modal_folder_items_content = "";
    project_modal_folder_items_content += project_modal_folder_items_content += "" +
        "<div class=\"row m-0 p-0\">" +
            "<div class=\"col-md-1 m-0 p-0\">" +
                "<button type=\"button\" class=\"btn btn-light btn-sm text-start w-100\" onclick=\"ProjectAddNewFolderAction(\'" + single_project_card_index + "\', \'" + response_folder + "\');\">" +
                    "<i class=\"bi bi-folder-plus\"></i>" +
                "</button>" +
            "</div>" +
            "<div class=\"col m-0 p-0\">" +
                "<input type=\"text\" class=\"form-control form-control-sm w-100\" id=\"project_add_new_folder_name_" + single_project_card_index + "\" placeholder=\"New folder name (default: New_Folder_0)\" style=\"overflow: auto\">" +
            "</div>" +
        "</div>";
    if (response_parent_folder != null)
    {
        project_modal_folder_items_content += "" +
            "<div class=\"row m-0 p-0\">" +
                "<button type=\"button\" class=\"col btn btn-light btn-sm text-start\" onclick=\"ProjectPathGetFolderItems(\'" + single_project_card_index + "\', \'" + response_parent_folder + "\');\">" +
                    "<i class=\"bi bi-folder\"></i> .." +
                "</button>" +
            "</div>";
    }
    for (let i = 0; i < response_folder_items.length; ++i)
    {
        if (response_folder_items[i]["type"] == "dir")
        {
            project_modal_folder_items_content += "" +
                "<div class=\"row m-0 p-0\">" +
                    "<button type=\"button\" class=\"col btn btn-light btn-sm text-start\" onclick=\"ProjectPathGetFolderItems(\'" + single_project_card_index + "\', \'" + response_folder + "/" + response_folder_items[i]['name'] + "\');\">" +
                        "<i class=\"bi bi-folder\"></i> " + response_folder_items[i]['name'] +
                    "</button>" +
                    "<button type=\"button\" class=\"col-md-1 btn btn-light btn-sm text-center\"></button>" +
                    "<button type=\"button\" class=\"col-md-1 btn btn-danger btn-sm text-center\" onclick=\"ProjectDeleteFolderAction(\'" + single_project_card_index + "\', \'" + response_folder + "\', \'" + response_folder_items[i]['name'] + "\');\">" +
                        "<i class=\"bi bi-trash\"></i>" +
                    "</button>" +
                "</div>";
        }
        else
        {
            project_modal_folder_items_content += "" +
                "<div class=\"row m-0 p-0\">" +
                    "<button type=\"button\" class=\"col btn btn-light btn-sm text-start\" disabled>" +
                        "<i class=\"bi bi-file-earmark\"></i> " + response_folder_items[i]['name'] +
                    "</button>" +
                    "<button type=\"button\" class=\"col-md-1 btn btn-light btn-sm text-center\"></button>" +
                    "<button type=\"button\" class=\"col-md-1 btn btn-danger btn-sm text-center\" onclick=\"ProjectDeleteFolderAction(\'" + single_project_card_index + "\', \'" + response_folder + "\', \'" + response_folder_items[i]['name'] + "\');\">" +
                        "<i class=\"bi bi-trash\"></i>" +
                    "</button>" +
                "</div>";
        }
    }
    document.getElementById("project_modal_folder_name_" + single_project_card_index).value = response_folder;
    document.getElementById("project_modal_folder_items_" + single_project_card_index).innerHTML = project_modal_folder_items_content;
    document.getElementById("project_modal_bottons_" + single_project_card_index).innerHTML = "<button type=\"button\" class=\"btn btn-primary btn-sm\" data-bs-dismiss=\"modal\" onclick=\"ProjectPathSaveButtonAction(\'" + single_project_card_index + "\', \'" + response_folder + "\');\">Save</button>";
});

// project dir save button, handle server response
socket.on("js_project_dir_save_button_action", (res) => {
    var single_project_card_index = res["single_project_card_index"];
    var response_single_project_card_dict = res["response_single_project_card_dict"];
    var if_path_correct = res["if_path_correct"];
    var if_first_load = res["if_first_load"];

    var project_card_dict = GetSessionStorageJsonItem("project_card_dict");
    if (if_path_correct)
    {
        project_card_dict[single_project_card_index] = response_single_project_card_dict;
        project_card_dict[single_project_card_index]["parameters_part_safe_flag"] = true;
        project_card_dict[single_project_card_index]["non_disabled_part"] = "parameters";
        SetSessionStorageJsonItem("project_card_dict", project_card_dict);
        UpdateProjectCard(single_project_card_index);
        UpdateInputCard();
    }
    else
    {
        project_card_dict[single_project_card_index] = response_single_project_card_dict;
        project_card_dict[single_project_card_index]["parameters_part_safe_flag"] = false;
        project_card_dict[single_project_card_index]["non_disabled_part"] = "project_dir";
        SetSessionStorageJsonItem("project_card_dict", project_card_dict);
        UpdateProjectCard(single_project_card_index);
        UpdateInputCard();
        if (!if_first_load)
        {
            alert("Illegal Path! Please choose a correct path");
        }
    }
});

// parameters save button, handle server response
socket.on("js_parameters_save_button_action", (res) => {
    var single_project_card_index = res["single_project_card_index"];
    var response_single_project_card_dict = res["response_single_project_card_dict"];

    var project_card_dict = GetSessionStorageJsonItem("project_card_dict");
    project_card_dict[single_project_card_index] = response_single_project_card_dict;
    project_card_dict[single_project_card_index]["results_panel_part_safe_flag"] = true;
    SetSessionStorageJsonItem("project_card_dict", project_card_dict);
    UpdateProjectCard(single_project_card_index);
    UpdateInputCard();
});

// result panel output panel, to show data
socket.on("js_result_panel_output_panel_show_data", (res) => {
    var single_project_card_index = res["single_project_card_index"];
    var response_text_list = res["data"];

    var response_text_html_value = "";
    for (let i = 0; i < response_text_list.length; ++i)
    {
        response_text_html_value += response_text_list[i] + "<br>";
    }
    document.getElementById(single_project_card_index + "_output_panel").innerHTML = response_text_html_value;
});

// alert cryowizard server message
socket.on("js_alert_message", (res) => {
    var single_project_card_index = res["single_project_card_index"];
    var server_message = res["message"];
    alert(single_project_card_index + ": " + server_message);
});
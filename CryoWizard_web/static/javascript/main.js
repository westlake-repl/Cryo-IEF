

function startimportandrefine()
{
    this.onclick = null;
    var StartImportAndRefine_socket = new EventSource("/ImportAndRefine");
    StartImportAndRefine_socket.onmessage = function (event)
    {
        if (event.data == "my_command_completed")
        {
            prior_info = document.getElementById("ImportAndRefine_output").innerHTML;
            document.getElementById("ImportAndRefine_output").innerHTML = prior_info + "[Complete!]<br>";
            StartImportAndRefine_socket.close();
        }
        else
        {
            prior_info = document.getElementById("ImportAndRefine_output").innerHTML;
            document.getElementById("ImportAndRefine_output").innerHTML = prior_info + event.data + "<br>";
        }
    }
}

// function startimport()
// {
//     var StartImport_socket = new EventSource("/Import");
//     StartImport_socket.onmessage = function (event)
//     {
//         if (event.data == "my_command_completed")
//         {
//             prior_info = document.getElementById("ImportAndRefine_output").innerHTML;
//             document.getElementById("ImportAndRefine_output").innerHTML = prior_info + "[Complete!]<br>";
//             StartImport_socket.close();
//         }
//         else
//         {
//             prior_info = document.getElementById("ImportAndRefine_output").innerHTML;
//             document.getElementById("ImportAndRefine_output").innerHTML = prior_info + event.data + "<br>";
//         }
//     }
// }
//
// function startrefine()
// {
//     var StartRefine_socket = new EventSource("/Refine");
//     StartRefine_socket.onmessage = function (event)
//     {
//         if (event.data == "my_command_completed")
//         {
//             prior_info = document.getElementById("ImportAndRefine_output").innerHTML;
//             document.getElementById("ImportAndRefine_output").innerHTML = prior_info + "[Complete!]<br>";
//             StartRefine_socket.close();
//         }
//         else
//         {
//             prior_info = document.getElementById("ImportAndRefine_output").innerHTML;
//             document.getElementById("ImportAndRefine_output").innerHTML = prior_info + event.data + "<br>";
//         }
//     }
// }

function killmyjob()
{
    this.onclick = null;
    var KillJob_socket = new EventSource("/KillJob");
    KillJob_socket.onmessage = function (event)
    {
        if (event.data == "my_command_completed")
        {
            prior_info = document.getElementById("ImportAndRefine_output").innerHTML;
            document.getElementById("ImportAndRefine_output").innerHTML = prior_info + "[Complete!]<br>";
            KillJob_socket.close();
        }
        else
        {
            prior_info = document.getElementById("ImportAndRefine_output").innerHTML;
            document.getElementById("ImportAndRefine_output").innerHTML = prior_info + event.data + "<br>";
        }
    }
}

function reconnect()
{
    this.onclick = null;
    var Reconnect_socket = new EventSource("/Reconnect");
    Reconnect_socket.onmessage = function (event)
    {
        if (event.data == "my_command_completed")
        {
            prior_info = document.getElementById("ImportAndRefine_output").innerHTML;
            document.getElementById("ImportAndRefine_output").innerHTML = prior_info + "[Complete!]<br>";
            Reconnect_socket.close();
        }
        else
        {
            prior_info = document.getElementById("ImportAndRefine_output").innerHTML;
            document.getElementById("ImportAndRefine_output").innerHTML = prior_info + event.data + "<br>";
        }
    }
}

function downloadmap()
{
    this.onclick = null;
    window.location.href = '/DownloadMap';
}
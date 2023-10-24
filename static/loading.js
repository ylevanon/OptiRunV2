// loading.js
document.addEventListener('DOMContentLoaded', function () {
    // Extracting the task_id from the URL
    var task_id = window.location.pathname.split('/').pop();

    // Make an AJAX request to check the status
    checkTaskStatus(task_id);
});

function checkTaskStatus(task_id) {
    $.ajax({
        url: "/check_status/" + task_id,
        type: "GET",
        success: function (data) {
            $("#status").text(data.status);
            if (data.status === "SUCCESS") {
                window.location.href = "/results/" + task_id;
            } else {
                setTimeout(function () {
                    checkTaskStatus(task_id);
                }, 5000); // Check every 5 seconds
            }
        },
        error: function (xhr, status, error) {
            console.error("Error:", error);
            // Handle the error as needed
        }
    });
}

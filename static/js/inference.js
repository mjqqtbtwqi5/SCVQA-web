$(document).ready(function () {
  let timerInterval;
  function pad(number) {
    return number.toString().padStart(2, "0");
  }
  $("#videoAssessmentForm").submit(function (e) {
    e.preventDefault();
    $("#videoAssessmentButton").prop("disabled", true);
    $("#videoAssessmentScoreSCVD").addClass("placeholder");
    $("#videoAssessmentScoreCSCVQ").addClass("placeholder");

    let form = $(this);
    let formData = new FormData(form[0]);

    let seconds = 0;
    let minutes = 0;
    let hours = 0;
    $("#timer").html("00:00:00");
    timerInterval = setInterval(() => {
      seconds++;
      if (seconds === 60) {
        seconds = 0;
        minutes++;
        if (minutes === 60) {
          minutes = 0;
          hours++;
        }
      }

      $("#timer").html(`${pad(hours)}:${pad(minutes)}:${pad(seconds)}`);
    }, 1000);

    $.ajax({
      type: "POST",
      url: "videoAssessment",
      data: formData,
      processData: false,
      contentType: false,
      success: function (response) {
        console.log(response);
        $("#videoAssessmentScoreSCVD").html(response.qualityScoreSCVD);
        $("#videoAssessmentScoreCSCVQ").html(response.qualityScoreCSCVQ);
      },
      error: function (xhr, status, error) {
        alert(error);
      },
      complete: function (msg) {
        $("#videoAssessmentScoreSCVD").removeClass("placeholder");
        $("#videoAssessmentScoreCSCVQ").removeClass("placeholder");
        $("#videoAssessmentButton").prop("disabled", false);
        clearInterval(timerInterval);
      },
    });
  });

  $(".featureAssessmentForm").submit(function (e) {
    e.preventDefault();

    let form = $(this);
    let formData = new FormData(form[0]);

    $.ajax({
      type: "POST",
      url: "featureAssessment",
      data: formData,
      processData: false,
      contentType: false,
      success: function (response) {
        form.find(".predict_mos_SCVD").html(response.qualityScoreSCVD);
        form.find(".predict_mos_CSCVQ").html(response.qualityScoreCSCVQ);
      },
      error: function (xhr, status, error) {
        console.log(error);
      },
    });
  });
});

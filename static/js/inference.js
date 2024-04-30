$(document).ready(function () {
  $("#videoAssessmentForm").submit(function (e) {
    e.preventDefault();
    $("#videoAssessmentButton").prop("disabled", true);
    $("#videoAssessmentScoreSCVD").addClass("placeholder");
    $("#videoAssessmentScoreCSCVQ").addClass("placeholder");

    let form = $(this);
    let formData = new FormData(form[0]);

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
        // Handle errors
        console.log(error);
      },
    });
  });
});

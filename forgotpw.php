<?php
// Initialize the session
session_start();

// Connection
require('connection.php');

// Vraiable for password
$new_password = $confirm_password = "";
$new_password_error = $confirm_password_error = "";

// Clean input variables
function test_input($data) {
  $data = trim($data);
  $data = stripslashes($data);
  $data = htmlspecialchars($data);
  return $data;
}

// Processing form data when form is submitted
if ($_SERVER["REQUEST_METHOD"] == "POST") {
  // Verify new password
  if(empty(test_input($_POST["new_password"]))) {
    $new_password_error = "Please enter a new password.";
    echo $new_password_error;
  }
  elseif (strlen(test_input($_POST["new_password"])) < 3) {
    $new_password_error = "Please enter a new password at least 3 characters long.";
    echo $new_password_error;
  }
  else {
    $new_password = test_input($_POST["new_password"]);
  }

  // Validate confirm password
  if(empty(test_input($_POST["confirm_password"]))) {
    $confirm_password_error = "Please enter a new confirmation password.";
    echo $confirm_password_error;
  }
  else {
    $confirm_password = test_input($_POST["confirm_password"]);

    if(empty($confirm_password_error) && ($new_password != $confirm_password)) {
      $confirm_password_error = "Password did not match.";
    }
  }

  // Validate password then update in database
  if(empty($confirm_password_error) && empty($new_password_error)) {
    // Prepare statement
    $sql = "UPDATE testschema.user_signup_log
            SET password = ?
            WHERE id = ?";
    // Bind parameters
    if($stmt = $conn->prepare($sql)) {
      $stmt->bind_param("si", $param_password, $param_id);

      // Set parameters
      $param_password = password_hash($new_password, PASSWORD_DEFAULT);
      $param_id = $_SESSION["id"];

      // Execute prepared statement
      if($stmt->execute()) {
        session_destroy();
        header("location: ../index.php");
        exit();
      }
      else {
        echo "There's an error. Please try again.";
      }
      $stmt->close();
    }
  }
}
?>

<!DOCTYPE html>
<html lang="en" dir="ltr">
  <head>
    <meta charset="utf-8">
    <title>Reset Password</title>
  </head>

  <link rel="stylesheet" href="../styles/webstyles.css">

  <body class="login_body">

    <div id="submit">
      <a href="../index.php">Back</a>
    </div>

    <div class="loginform1">
      <h2>Reset Password</h2>
        <p>Please fill out this form to reset your password.</p>
          <form action="<?php echo htmlspecialchars($_SERVER["PHP_SELF"]); ?>" method="post">
              <div class="form-group <?php echo (!empty($new_password_error)) ? 'has-error' : ''; ?>">
                  <label>New Password<br></label>
                  <input type="password" name="new_password" class="form-control" value="<?php echo $new_password; ?>">
                  <span class="help-block"><?php echo $new_password_error; ?></span>
              </div>
              <div class="form-group <?php echo (!empty($confirm_password_error)) ? 'has-error' : ''; ?>">
                  <label>Confirm Password<br></label>
                  <input type="password" name="confirm_password" class="form-control"><br><br>
                  <span class="help-block"><?php echo $confirm_password_error; ?></span>
              </div>
              <div id="submit">
                  <input type="submit" class="btn btn-primary" value="Submit">
              </div>
          </form>
      </div>
  </body>
</html>

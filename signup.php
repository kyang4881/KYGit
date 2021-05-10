<!DOCTYPE html>
<html lang="en" dir="ltr">
<!--<div style = "background-image: url('bday.jpg');">-->
  <head>
    <meta charset="utf-8">
    <title>Sign Up</title>
    <style>
      body {
        /*background-image: url("bday.jpg");
        background-size: cover;*/
      }
    </style>
  </head>

  <link rel="stylesheet" href="../styles/webstyles.css">

  <body class="login_body">

    <div id="submit">
      <a href="../index.php">Back</a>
    </div>

    <div class="loginform1">
      <h2>Sign Up</h2>
      <form method="post" action="<?php echo htmlspecialchars($_SERVER["PHP_SELF"]);?>">
        Login ID: <input type="text" name="login_id_input"><br>
        Password: <input type="password" name="password_input"><br>
        Confirm Password: <input type="password" name="password_confirm_input"><br>
        Email: <input type="text" name="email_input"><br>
        Phone #: <input type="text" name="phone_num_input"><br><br>
        <div id="submit"><input type="submit" name="submit" value="Submit"></div><br><br>
      </form>
    </div>

    <?php
    require('connection.php');
    //ob_start();
    function test_input($data) {
      $data = trim($data);
      $data = stripslashes($data);
      $data = htmlspecialchars($data);
      return $data;
    }

    // Create variables
    $login_id_input = $password_input = $password_confirm_input = $email_input = $phone_num_input = "";
    $login_id_input_error = $password_input_error = $password_confirm_error = $email_input = $phone_num_input = "";

    if ($_SERVER["REQUEST_METHOD"] == "POST") {
      // Clean inputs
      $login_id_input = test_input($_POST["login_id_input"]);
      $password_input = test_input($_POST["password_input"]);
      $password_confirm_input = test_input($_POST["password_confirm_input"]);
      $email_input = test_input($_POST["email_input"]);
      $phone_num_input = test_input($_POST["phone_num_input"]);

      // Validate login id
      if(empty($login_id_input)) {
        $login_id_input_error = "Please enter a login id.<br>";
        echo $login_id_input_error;
      }

      else {
        // Prepare a select statement
        $sql = "SELECT login_id
                FROM testschema.user_signup_log
                WHERE login_id = ?";

        if($stmt = $conn->prepare($sql)) {
          // Bind variables to prepared statement as parameters
          $stmt->bind_param("s", $param_login_id);
          // Set parameters
          $param_login_id = $login_id_input;
          // Attempt to execute the prepared statement
          if($stmt->execute()) {
            // Store result
            $stmt->store_result();

            if($stmt->num_rows == 1) {
              $login_id_input_error = "Login id already taken.<br>";
              echo $login_id_input_error;
            }
          }

          else {
            echo "Please try again later.<br>";
          }
          // Close statement
          $stmt->close();
        }
      }

      // Validate password
      if(empty($password_input)) {
        $password_input_error = "Please enter a password.<br>";
        echo $password_input_error;
      }
      elseif(strlen($password_input) < 3) {
        $password_input_error = "Please enter a password with atleast 3 characters. <br>";
        echo $password_input_error;
      }

      // Validate confirm password
      if(empty($password_confirm_input)) {
        $password_confirm_error = "Please enter a confirmation password.<br>";
        echo $password_confirm_error;
      }
      else {
        if(empty($password_confirm_error) && ($password_input != $password_confirm_input)) {
          $password_confirm_error = "Confirmation password did not match.<br>";
          echo $password_confirm_error;
        }
      }

      // Display input values (Optional)
      echo $login_id_input . "<br>";
      echo password_hash($password_input, PASSWORD_DEFAULT) . "<br>";
      echo password_hash($password_confirm_input, PASSWORD_DEFAULT) . "<br>";
      echo $email_input . "<br>";
      echo $phone_num_input . "<br>";

      // Check input errors before inserting in database
      if(empty($login_id_input_error) && empty($password_input_error) && empty($password_confirm_error)) {
        // Prepare an insert statement
        $sql = "INSERT INTO testschema.user_signup_log (login_id, password, confirm_password, email, phone_num) VALUES (?, ?, ?, ?, ?)";

        if($stmt = $conn->prepare($sql)) {
          // Bind variables to the prepared statement as parameters
          $stmt->bind_param("sssss", $param_login_id,
                                     $param_password,
                                     $param_confirm_password,
                                     $param_email,
                                     $param_phone_num);

          $param_login_id = $login_id_input;
          $param_password = password_hash($password_input, PASSWORD_DEFAULT);
          $param_confirm_password = password_hash($password_confirm_input, PASSWORD_DEFAULT);
          $param_email = $email_input;
          $param_phone_num = $phone_num_input;

          // Attempt to execute the prepared statement
          if($stmt->execute()) {
            header("location: ../index.php"); //Important Note: A header warning will result of the sql user signup log is displayed before this line of code (the black of php codes below)
            die();
            //echo("<script>location.href = '/index.php';</script>");
          }
          else {
            echo "Something went wrong. Please try again later.<br>";
          }
          $stmt->close();
        }
      }
    $conn->close();
    }
    //ob_end_flush();
    ?>

    <?php //require('db_tables.php') ?>

  </body>
</html>

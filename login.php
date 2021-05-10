<?php
// Initialize session
session_start();

// Check if the user is already logged in, if yes then redirect him to welcome page
if(isset($_SESSION["logged_in"]) && $_SESSION["logged_in"] === True) {
  header("location: ../scripts/loggedin.php");
  exit;
}

// Connect to DB
require('connection.php');

// Clean input variables
function test_input($data) {
  $data = trim($data);
  $data = stripslashes($data);
  $data = htmlspecialchars($data);
  return $data;
}

// Processing form data when form is submitted
if ($_SERVER["REQUEST_METHOD"] == "POST") {
  // Create variables
  $login_id_input = $password_input = "";
  $login_id_input_error = $password_input_error = "";

  $login_id_input = test_input($_POST["login_id_input"]);
  $password_input = test_input($_POST["password_input"]);

  // Display input values (Optional)
  //echo "<h2> Your Inputs: </h2>";
  //echo "your login_id is: " . $login_id_input . "<br>";
  //echo "your password is: " . $password_input . "<br><br>";

  // Check if login id is empty
  if (empty($login_id_input)) {
    $login_id_input_error = "Please enter your login id.<br>";
    echo $login_id_input_error;
  }

  // Check if password is empty
  if (empty($password_input)) {
    $password_input_error = "Please enter your password.<br>";
    echo $password_input_error;
  }

  // If log id and password are entered, validate credientials
  if (empty($login_id_input_error) && empty($password_input_error)) {
    // Prepare statement
    $sql = "SELECT id, login_id, password
            FROM testschema.user_signup_log
            WHERE login_id = ?";

    if ($stmt = $conn->prepare($sql)) {
      // Bind variables to prepared statement
      $stmt->bind_param("s", $param_login_id);

      // Set binded variables
      $param_login_id = $login_id_input;

      // Attempt to execute prepared statement
      if($stmt->execute()){
        // Store results
        $stmt->store_result();

        // Check if login id exists in the user_signup_log
        if($stmt->num_rows == 1) {
          // Bind result variables
          $stmt->bind_result($id, $login_id_input, $hashed_password);

          if($stmt->fetch()) {
            if(password_verify($password_input, $hashed_password)) {
              // Password is correct, start a new session
              session_start();

              // Store data in session variables
              $_SESSION["loggedin"] = true;
              $_SESSION["id"] = $id;
              $_SESSION["login_id_input"] = $login_id_input;

              // Redirect user to logged in page
              header("location: ../scripts/loggedin.php");
              exit();
            }
            else {
              // Password incorrect
              $password_input_error = "Password incorrect.<br>";
              echo $password_input_error;
            }
          }
        }
        else {
          $login_id_input_error = "Login id not found.<br>";
          echo $login_id_input_error;
        }
      }
      else {
        echo "There's an error. Please try again.<br>";
      }

    // Close statement
    $stmt->close();
    }
  }
}

?>

<?php
if ($_SERVER["REQUEST_METHOD"] == "POST") {
  // Prepare and Bind
  $stmt = $conn->prepare("INSERT INTO testschema.user_login_log (login_id, password, login_error, password_error)
                          VALUES (?,?,?,?)");
  $stmt->bind_param("ssss", $login_id, $password, $login_error, $password_error);

  // Set parameters and execute
  // Only log id and pw if there are entries in both
  if (!empty($login_id_input) and !empty($password_input)) {
    $login_id = $login_id_input;
    $password = $password_input;
    $login_error = $login_id_input_error;
    $password_error = $password_input_error;
    $stmt->execute();

    //echo "<br> New records created successfully to table testschema.user_login_log";
  }
  $stmt->close();
}
?>

<!DOCTYPE html>
<html lang="en" dir="ltr">
  <head>
    <meta charset="utf-8" name = "viewport" content="width=device-width, initial-scale=1.0">
    <title>Login</title>
  </head>

  <link rel="stylesheet" href="../styles/webstyles.css">

  <body class="login_body">

    <div id="submit">
      <a href="../index.php">Back</a>
    </div>

    <div class="loginform1">
      <h2>Login</h2>
        <form method="post" action="<?php echo htmlspecialchars($_SERVER["PHP_SELF"]); ?>">
          Login ID : <input type="text" name="login_id_input"><br>
          Password: <input type="password" name="password_input"><br><br>
          <div id="submit" ><input id= "submit" type="submit" name="submit" value="Submit"><br><br></div>
          <a id="login_reset" href="forgotpw.php">Forgot Password</a>
          <a id="login_signup" href="signup.php">Sign Up</a>
          <a id="login_logout" href="logout.php">Log Out</a>
        </form>
    </div>
  </body>
</html>

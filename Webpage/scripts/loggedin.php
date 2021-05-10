<?php
  session_start();

  if(!isset($_SESSION["loggedin"]) || $_SESSION["loggedin"] !== true){
    header("location: ../index.php");
  exit;
}
?>

<!DOCTYPE html>
<html lang="en" dir="ltr">
  <head>
    <meta charset="utf-8">
    <title>Logged In</title>
    <style>
    </style>
  </head>

  <link rel="stylesheet" href="../styles/webstyles.css"> <!--for retrieving styles-->

  <body class="login_body">

    <div>
      <h1>
        <b><p style="color:black" align="center">Login Successful.</p></b>
        <p style="color:black" align="center">Welcome, <?php echo htmlspecialchars($_SESSION["login_id_input"]); ?>!
        </p>
      <h1>
    </div>

    <div class="loginform1">
      <a href="../index.php" class="btn btn-warning">Home Page</a></p>
      <a href="resetpw.php" class="btn btn-warning">Reset Password</a></p>
      <a href="logout.php" class="btn btn-danger">Log Out</a></p>
    </div>

  </body>
</html>

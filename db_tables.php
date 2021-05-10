<?php
require('connection.php');
// Display DB login Info Log Table
$sql = "SELECT * FROM testschema.user_login_log";
$result = $conn->query($sql);

if ($result->num_rows > 0) {

  echo "<br><br><br>
        <table id='t1'>
          <caption>From the Database: testschema.user_login_log</caption>
          <tr>
            <th>ID</th>
            <th>Login_id</th>
            <th>Password</th>
            <th>Login_error</th>
            <th>Password_error</th>
            <th>Reg_Date</th>
          </tr>";
  // output data of each row
  while($row = $result->fetch_assoc()) {
    echo "<tr>
            <td>".$row["id"]."</td>
            <td>".$row["login_id"]."</td>
            <td>".$row["password"]."</td>
            <td>".$row["Login_error"]."</td>
            <td>".$row["Password_error"]."</td>
            <td>".$row["reg_date"]."</td>
          </tr>";
  }
  echo "</table>";
} else {
  echo "0 results";
}
$conn->close();

?><br>

<?php
require('connection.php');
// Display DB login Info Log Table (Optional)
$sql = "SELECT * FROM testschema.user_signup_log";
$result = $conn->query($sql);

if ($result->num_rows > 0) {

  echo "<table id='t2'>
          <caption>From the Database: testschema.user_signup_log</caption>
          <tr>
            <th>id</th>
            <th>login_id</th>
            <th>password</th>
            <th>confirm_password</th>
            <th>email</th>
            <th>phone_num</th>
            <th>Reg_Date</th>
          </tr>";
  // output data of each row
  while($row = $result->fetch_assoc()) {
    echo "<tr>
            <td>".$row["id"]."</td>
            <td>".$row["login_id"]."</td>
            <td>".$row["password"]."</td>
            <td>".$row["confirm_password"]."</td>
            <td>".$row["email"]."</td>
            <td>".$row["phone_num"]."</td>
            <td>".$row["reg_date"]."</td>
          </tr>";
  }
  echo "</table>";
} else {
  echo "0 results";
}
?><br>

<?php
require('connection.php');
// Display DB login Info Log Table (Optional)
$sql = "SELECT * FROM testschema.user_signup_log";
$result = $conn->query($sql);

if ($result->num_rows > 0) {
  echo "<h2>From the Database: testschema.user_signup_log</h2>";

  echo "<table>
          <tr>
            <th>id</th>
            <th>login_id</th>
            <th>password</th>
            <th>confirm_password</th>
            <th>email</th>
            <th>phone_num</th>
            <th>Reg_Date</th>
          </tr>";
  // output data of each row
  while($row = $result->fetch_assoc()) {
    echo "<tr>
            <td>".$row["id"]."</td>
            <td>".$row["login_id"]."</td>
            <td>".$row["password"]."</td>
            <td>".$row["confirm_password"]."</td>
            <td>".$row["email"]."</td>
            <td>".$row["phone_num"]."</td>
            <td>".$row["reg_date"]."</td>
          </tr>";
  }
  echo "</table>";
} else {
  echo "0 results";
}
?><br>

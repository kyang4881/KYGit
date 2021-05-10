<?php
// Connection PHP Script
require('connection.php');

// Create Database
$sql = "CREATE DATABASE kDB";
if ($conn->query($sql) === TRUE) {
  echo "Database created successfully";
} else {
  echo "Error creating database: " . $conn->error;
}

// Create table to store user login info
$sql = "CREATE TABLE testschema.user_login_log  (
         id INT(6) UNSIGNED AUTO_INCREMENT PRIMARY KEY,
         login_id VARCHAR(30) NOT NULL,
         password VARCHAR(30) NOT NULL,
         reg_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
       )";

echo "<br>";

if ($conn->query($sql) === TRUE) {
  echo "Table user_login_log created successfully";
} else {
  echo "Error creating table: " . $conn->error;
}

$conn->close();
?>

<?php
// Connection PHP Script
require('connection.php');

// Create table to store user signup info
$sql = "CREATE TABLE testschema.user_signup_log  (
         id INT(6) UNSIGNED AUTO_INCREMENT PRIMARY KEY,
         login_id VARCHAR(30) NOT NULL,
         password VARCHAR(30) NOT NULL,
         confirm_password VARCHAR(30) NOT NULL,
         email VARCHAR(50) NOT NULL,
         phone_num INT(11) NOT NULL,
         reg_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
       )";

echo "<br>";

if ($conn->query($sql) === TRUE) {
  echo "Table testschema.user_signup_log created successfully";
} else {
  echo "Error creating table: " . $conn->error;
}

$conn->close();
?>

<?php
// Connection PHP Script
require('connection.php');

// Create table to store user signup info
$sql = "CREATE TABLE testschema.store_inventory  (
         id INT(6) UNSIGNED AUTO_INCREMENT PRIMARY KEY,
         item_name VARCHAR(30) NOT NULL,
         item_price int(30) NOT NULL,
         item_qty VARCHAR(30) NOT NULL,
         reg_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
       )";

echo "<br>";

if ($conn->query($sql) === TRUE) {
  echo "Table testschema.store_inventory created successfully";
} else {
  echo "Error creating table: " . $conn->error;
}

$conn->close();
?>

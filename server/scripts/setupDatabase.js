const sqlite3 = require('sqlite3').verbose();
const path = require('path');

const dbPath = path.join(__dirname, '../database.sqlite');
const db = new sqlite3.Database(dbPath);

const createTables = () => {
  return new Promise((resolve, reject) => {
    db.serialize(() => {
      // Users table
      db.run(`
        CREATE TABLE IF NOT EXISTS users (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          username VARCHAR(50) UNIQUE NOT NULL,
          email VARCHAR(100) UNIQUE NOT NULL,
          password_hash VARCHAR(255) NOT NULL,
          name VARCHAR(100) NOT NULL,
          location VARCHAR(100),
          profile_photo VARCHAR(255),
          bio TEXT,
          availability TEXT,
          is_public BOOLEAN DEFAULT 1,
          is_banned BOOLEAN DEFAULT 0,
          is_admin BOOLEAN DEFAULT 0,
          created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
          updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
      `);

      // Skills table
      db.run(`
        CREATE TABLE IF NOT EXISTS skills (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          name VARCHAR(100) NOT NULL,
          category VARCHAR(50),
          description TEXT,
          is_approved BOOLEAN DEFAULT 1,
          created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
      `);

      // User offered skills
      db.run(`
        CREATE TABLE IF NOT EXISTS user_offered_skills (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          user_id INTEGER,
          skill_id INTEGER,
          proficiency_level VARCHAR(20) DEFAULT 'intermediate',
          description TEXT,
          created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
          FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
          FOREIGN KEY (skill_id) REFERENCES skills(id) ON DELETE CASCADE
        )
      `);

      // User wanted skills
      db.run(`
        CREATE TABLE IF NOT EXISTS user_wanted_skills (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          user_id INTEGER,
          skill_id INTEGER,
          urgency VARCHAR(20) DEFAULT 'medium',
          description TEXT,
          created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
          FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
          FOREIGN KEY (skill_id) REFERENCES skills(id) ON DELETE CASCADE
        )
      `);

      // Swap requests
      db.run(`
        CREATE TABLE IF NOT EXISTS swap_requests (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          requester_id INTEGER,
          provider_id INTEGER,
          requested_skill_id INTEGER,
          offered_skill_id INTEGER,
          message TEXT,
          status VARCHAR(20) DEFAULT 'pending',
          created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
          updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
          FOREIGN KEY (requester_id) REFERENCES users(id) ON DELETE CASCADE,
          FOREIGN KEY (provider_id) REFERENCES users(id) ON DELETE CASCADE,
          FOREIGN KEY (requested_skill_id) REFERENCES skills(id),
          FOREIGN KEY (offered_skill_id) REFERENCES skills(id)
        )
      `);

      // Ratings and feedback
      db.run(`
        CREATE TABLE IF NOT EXISTS ratings (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          swap_request_id INTEGER,
          rater_id INTEGER,
          rated_id INTEGER,
          rating INTEGER CHECK(rating >= 1 AND rating <= 5),
          feedback TEXT,
          created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
          FOREIGN KEY (swap_request_id) REFERENCES swap_requests(id) ON DELETE CASCADE,
          FOREIGN KEY (rater_id) REFERENCES users(id) ON DELETE CASCADE,
          FOREIGN KEY (rated_id) REFERENCES users(id) ON DELETE CASCADE
        )
      `);

      // Admin messages
      db.run(`
        CREATE TABLE IF NOT EXISTS admin_messages (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          admin_id INTEGER,
          title VARCHAR(200) NOT NULL,
          content TEXT NOT NULL,
          is_active BOOLEAN DEFAULT 1,
          created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
          FOREIGN KEY (admin_id) REFERENCES users(id) ON DELETE CASCADE
        )
      `);

      // Admin actions log
      db.run(`
        CREATE TABLE IF NOT EXISTS admin_actions (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          admin_id INTEGER,
          action_type VARCHAR(50) NOT NULL,
          target_type VARCHAR(50),
          target_id INTEGER,
          description TEXT,
          created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
          FOREIGN KEY (admin_id) REFERENCES users(id) ON DELETE CASCADE
        )
      `);

      // Insert some default skills
      const defaultSkills = [
        ['Programming', 'Technology', 'General programming skills'],
        ['JavaScript', 'Technology', 'JavaScript programming language'],
        ['Python', 'Technology', 'Python programming language'],
        ['React', 'Technology', 'React JavaScript library'],
        ['Node.js', 'Technology', 'Node.js runtime environment'],
        ['Photoshop', 'Design', 'Adobe Photoshop image editing'],
        ['Excel', 'Business', 'Microsoft Excel spreadsheet software'],
        ['Writing', 'Creative', 'Creative and technical writing'],
        ['Music Production', 'Creative', 'Audio production and mixing'],
        ['Guitar', 'Music', 'Guitar playing and instruction'],
        ['Cooking', 'Life Skills', 'Cooking and culinary skills'],
        ['Fitness Training', 'Health', 'Personal fitness and training'],
        ['Language Exchange', 'Education', 'Language learning and teaching'],
        ['Marketing', 'Business', 'Digital and traditional marketing'],
        ['Data Analysis', 'Technology', 'Data analysis and visualization']
      ];

      const insertSkill = db.prepare('INSERT OR IGNORE INTO skills (name, category, description) VALUES (?, ?, ?)');
      defaultSkills.forEach(skill => {
        insertSkill.run(skill);
      });
      insertSkill.finalize();

      // Create admin user
      const bcrypt = require('bcryptjs');
      const adminPassword = bcrypt.hashSync('admin123', 10);
      
      db.run(`
        INSERT OR IGNORE INTO users (username, email, password_hash, name, is_admin) 
        VALUES ('admin', 'admin@skillswap.com', ?, 'System Admin', 1)
      `, [adminPassword]);

      resolve();
    });
  });
};

createTables()
  .then(() => {
    console.log('✅ Database setup completed successfully!');
    console.log('📊 Created tables: users, skills, user_offered_skills, user_wanted_skills, swap_requests, ratings, admin_messages, admin_actions');
    console.log('🔑 Default admin user created: admin@skillswap.com / admin123');
    db.close();
  })
  .catch((err) => {
    console.error('❌ Database setup failed:', err);
    db.close();
  });
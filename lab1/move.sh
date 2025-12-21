#!/bin/bash

# Скрипт для переноса веток lab1, lab2, lab3, lab4 в папки
# Убедитесь, что вы в корне репозитория

# Получаем все ветки с remote
echo "Fetching branches from remote..."
git fetch veekay/ke1thuzad

# Переключаемся на main и обновляем ее
echo "Switching to main branch..."
git checkout main
git pull origin main

# Обрабатываем каждую ветку
for lab_num in {1..4}; do
    branch_name="lab${lab_num}"
    folder_name="lab${lab_num}"
    
    echo "Processing branch: ${branch_name}..."
    
    # Проверяем, существует ли ветка на remote
    if ! git show-ref --verify --quiet "refs/remotes/ke1thuzad/${branch_name}"; then
        echo "Warning: Branch ${branch_name} not found in remote veekay/ke1thuzad"
        continue
    fi
    
    # Создаем временную ветку из remote ветки
    echo "Creating temporary branch from ${branch_name}..."
    git checkout -b "temp_${branch_name}" "ke1thuzad/${branch_name}"
    
    # Создаем папку для этой ветки
    mkdir -p "${folder_name}"
    
    # Перемещаем все файлы (кроме .git и самой папки) в новую папку
    echo "Moving files to ${folder_name}/..."
    
    # Вариант 1: Простое перемещение всех файлов и папок
    # Исключаем .git, саму созданную папку и возможные скрытые файлы .git*
    find . -maxdepth 1 \
        -not -name "." \
        -not -name ".git" \
        -not -name ".git*" \
        -not -name "${folder_name}" \
        -exec mv {} "${folder_name}/" 2>/dev/null \; || true
    
    # Добавляем изменения
    git add .
    
    # Проверяем, есть ли изменения для коммита
    if ! git diff-index --quiet HEAD --; then
        git commit -m "Move ${branch_name} to ${folder_name}/ folder"
    else
        echo "No changes to commit for ${branch_name}"
    fi
    
    # Возвращаемся в main и делаем merge
    echo "Merging into main..."
    git checkout main
    git merge --no-ff "temp_${branch_name}" -m "Merge ${branch_name} into ${folder_name}/ folder"
    
    # Удаляем временную ветку
    git branch -d "temp_${branch_name}"
    
    echo "Completed: ${branch_name} -> ${folder_name}/"
    echo "---"
done

echo "All branches have been processed!"
echo "Check the result with: ls -la"
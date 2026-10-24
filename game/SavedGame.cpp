#include "SavedGame.h"
#include <string.h>

bool SavedGames_IsValidSessionImage(const void *buffer, size_t size)
{
    if (!buffer || (size != sizeof(SavedGameImage) && size != sizeof(SavedGameFileImage)))
        return false;
    const SavedGameImage &image = *(const SavedGameImage *)buffer;
    if (image.options.playerCount < 1 || image.options.playerCount > 4 ||
        image.options.mode > 10 || image.options.difficulty > 2 ||
        image.stage.country >= 8 || image.stage.stage >= 11 ||
        !memchr(image.name, 0, sizeof(image.name)))
        return false;
    for (unsigned i = 0; i < image.options.playerCount; i++)
        if (!memchr(image.playerIdentifiers[i].name, 0, sizeof(image.playerIdentifiers[i].name)))
            return false;
    if (size == sizeof(SavedGameFileImage)) {
        const SavedGameServiceState &service = ((const SavedGameFileImage *)buffer)->service;
        if (service.magic != SAVED_GAME_SERVICE_MAGIC ||
            service.version != SAVED_GAME_SERVICE_VERSION ||
            service.size != sizeof(service))
            return false;
        for (unsigned player = 0; player < image.options.playerCount; player++) {
            for (unsigned i = 0; i < 12; i++)
                if (service.repairFlags[player][i] > 1)
                    return false;
            for (unsigned i = 0; i < 7; i++)
                if (service.setupFlags[player][i] > 1)
                    return false;
        }
    }
    return true;
}
